import os
import urllib.request
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, AutoModel, BitsAndBytesConfig
import torch
import requests
import cv2
from PIL import Image
import warnings
from transformers import logging
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import pandas as pd
from abc import ABC, abstractmethod
from transformers import AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from transformers import AlignProcessor, AlignModel

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Load face detector once to avoid per-image overhead
try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    FACE_CASCADE = None

# ============================================================================
# Vision Model Adapter Architecture
# ============================================================================

class VisionModelAdapter(ABC):
    """Abstract base class for vision model adapters.
    
    This class provides a standard interface for different vision model architectures,
    allowing users to extend transforEmotion with new model types beyond CLIP.
    """
    
    def __init__(self, model_id, local_model_path=None):
        self.model_id = model_id
        self.local_model_path = local_model_path
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False
    
    @abstractmethod
    def load_model(self):
        """Load the model and processor/tokenizer components."""
        pass
    
    @abstractmethod
    def process_image(self, image):
        """Process image according to model requirements.
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Processed image inputs ready for model
        """
        pass
    
    @abstractmethod
    def process_text(self, labels):
        """Process text labels according to model requirements.
        
        Args:
            labels: List of text labels
            
        Returns:
            dict: Processed text inputs ready for model
        """
        pass
    
    @abstractmethod
    def compute_similarities(self, image_inputs, text_inputs):
        """Compute similarities between image and text inputs.
        
        Args:
            image_inputs: Processed image inputs
            text_inputs: Processed text inputs
            
        Returns:
            torch.Tensor: Similarity scores/probabilities
        """
        pass
    
    def __call__(self, image, labels):
        """Main inference method.
        
        Args:
            image: PIL Image object
            labels: List of text labels
            
        Returns:
            dict: Label -> probability mapping
        """
        if not self._initialized:
            self.load_model()
            self._initialized = True
        
        with torch.no_grad():
            image_inputs = self.process_image(image)
            text_inputs = self.process_text(labels)
            probs = self.compute_similarities(image_inputs, text_inputs)
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return dict(zip(labels, probs.tolist()))


class CLIPAdapter(VisionModelAdapter):
    """Standard CLIP model adapter."""
    
    def load_model(self):
        """Load standard CLIP model and processor."""
        source_path = self.local_model_path if self.local_model_path else self.model_id
        source_type = "local directory" if self.local_model_path else "HuggingFace"
        print(f"Loading CLIP model from {source_type}: {source_path}")
        
        self.processor = CLIPProcessor.from_pretrained(
            source_path, 
            local_files_only=bool(self.local_model_path)
        )
        self.model = CLIPModel.from_pretrained(
            source_path, 
            ignore_mismatched_sizes=True,
            local_files_only=bool(self.local_model_path)
        )
        self.model = self.model.to(self.device)
    
    def process_image(self, image):
        """Process image with CLIP processor."""
        inputs = self.processor(images=image, return_tensors='pt')
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
               for k, v in inputs.items()}
    
    def process_text(self, labels):
        """Process text with CLIP processor."""
        inputs = self.processor(text=labels, return_tensors='pt', padding=True)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
               for k, v in inputs.items()}
    
    def compute_similarities(self, image_inputs, text_inputs):
        """Compute CLIP-style similarities."""
        image_embeds = self.model.get_image_features(**image_inputs)
        text_embeds = self.model.get_text_features(**text_inputs)
        
        # Normalize embeddings
        image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate similarity and probabilities
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * self.model.logit_scale.exp()
        probs = logits_per_image.softmax(dim=1).squeeze(0)
        
        return probs


class JinaCLIPAdapter(VisionModelAdapter):
    """Jina CLIP v2 model adapter with custom preprocessing."""
    
    def load_model(self):
        """Load Jina CLIP model with custom configuration."""
        source_path = self.local_model_path if self.local_model_path else self.model_id
        source_type = "local directory" if self.local_model_path else "HuggingFace"
        print(f"Loading Jina CLIP model from {source_type}: {source_path}")
        
        self.model = AutoModel.from_pretrained(
            source_path, 
            trust_remote_code=True,
            local_files_only=bool(self.local_model_path)
        )
        self.model = self.model.to(self.device)
        
        # Use standard tokenizer unless provided locally
        tokenizer_path = self.local_model_path if self.local_model_path else "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=bool(self.local_model_path)
        )
        
        # Define Jina-specific transform
        self.transform = T.Compose([
            T.Resize(512, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def process_image(self, image):
        """Process image with Jina-specific transform."""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return {'pixel_values': image_tensor}
    
    def process_text(self, labels):
        """Process text with tokenizer."""
        inputs = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
               for k, v in inputs.items()}
    
    def compute_similarities(self, image_inputs, text_inputs):
        """Compute Jina CLIP similarities."""
        image_embeds = self.model.get_image_features(**image_inputs)
        text_embeds = self.model.get_text_features(**text_inputs)
        
        # Normalize embeddings
        image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate similarity (Jina models may not have logit_scale)
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())
        if hasattr(self.model, 'logit_scale'):
            logits_per_image *= self.model.logit_scale.exp()
        
        probs = logits_per_image.softmax(dim=1).squeeze(0)
        return probs


class EVACLIPAdapter(VisionModelAdapter):
    """EVA CLIP adapter with quantization support."""
    
    def load_model(self):
        """Load EVA CLIP model with quantization if available."""
        source_path = self.local_model_path if self.local_model_path else self.model_id
        source_type = "local directory" if self.local_model_path else "HuggingFace"
        print(f"Loading EVA CLIP model from {source_type}: {source_path}")
        
        try:
            print("Attempting to load EVA-CLIP-8B model with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = CLIPModel.from_pretrained(
                source_path,
                ignore_mismatched_sizes=True,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                local_files_only=bool(self.local_model_path)
            )
            print("Successfully loaded EVA-CLIP-8B model with quantization")
        except Exception as e:
            print(f"Quantized model loading failed: {str(e)}")
            print("Falling back to standard loading method...")
            self.model = CLIPModel.from_pretrained(
                source_path,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float32,
                local_files_only=bool(self.local_model_path)
            )
            print("Successfully loaded EVA-CLIP-8B model without quantization.")
        
        self.model = self.model.to(self.device)
        
        # EVA-specific transform
        self.transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])
        
        # Standard tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    def process_image(self, image):
        """Process image with EVA-specific transform."""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return {'pixel_values': image_tensor}
    
    def process_text(self, labels):
        """Process text with tokenizer."""
        inputs = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
               for k, v in inputs.items()}
    
    def compute_similarities(self, image_inputs, text_inputs):
        """Compute EVA CLIP similarities."""
        image_embeds = self.model.get_image_features(**image_inputs)
        text_embeds = self.model.get_text_features(**text_inputs)
        
        # Normalize embeddings
        image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate similarity
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())
        if hasattr(self.model, 'logit_scale'):
            logits_per_image *= self.model.logit_scale.exp()
        
        probs = logits_per_image.softmax(dim=1).squeeze(0)
        return probs


class BLIPAdapter(VisionModelAdapter):
    """BLIP image-captioning adapter using negative log-likelihood scoring.

    Approach:
    - Load BLIP processor and conditional generation model.
    - For each candidate label (treated as target caption), compute the
      cross-entropy loss of generating that label conditioned on the image.
    - Convert negative losses into probabilities via softmax.
    """

    def load_model(self):
        source_path = self.local_model_path if self.local_model_path else self.model_id
        source_type = "local directory" if self.local_model_path else "HuggingFace"
        print(f"Loading BLIP model from {source_type}: {source_path}")

        # Processor + BLIP conditional generation model
        # Prefer explicit BlipProcessor; AutoProcessor works too but is less explicit
        try:
            self.processor = BlipProcessor.from_pretrained(
                source_path,
                local_files_only=bool(self.local_model_path)
            )
        except Exception:
            # Fallback to AutoProcessor if BlipProcessor is unavailable
            self.processor = AutoProcessor.from_pretrained(
                source_path,
                local_files_only=bool(self.local_model_path)
            )

        self.model = BlipForConditionalGeneration.from_pretrained(
            source_path,
            local_files_only=bool(self.local_model_path)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Ensure tokenizer has a PAD token for proper loss masking
        try:
            tok = getattr(self.processor, 'tokenizer', None)
            if tok is not None and tok.pad_token_id is None and getattr(tok, 'eos_token_id', None) is not None:
                tok.pad_token = tok.eos_token
        except Exception:
            pass

    def process_image(self, image):
        # Pre-tokenize pixel values only; labels will be handled separately
        inputs = self.processor(images=image, return_tensors="pt")
        # Ensure tensors are on device
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def process_text(self, labels):
        # Tokenize labels; used as targets for loss computation
        inputs = self.processor(text=labels, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def compute_similarities(self, image_inputs, text_inputs):
        # Compute per-label NLL by running the model with labels
        pixel_values = image_inputs.get('pixel_values')
        input_ids = text_inputs.get('input_ids')
        attention_mask = text_inputs.get('attention_mask')

        if pixel_values is None or input_ids is None:
            raise ValueError("BLIPAdapter requires pixel_values and input_ids")

        losses = []
        with torch.no_grad():
            # Loop over labels to obtain an individual loss per label
            for i in range(input_ids.shape[0]):
                label_ids = input_ids[i].unsqueeze(0).to(self.device)
                # Convert padding tokens to -100 so they are ignored in loss
                if attention_mask is not None:
                    label_mask = attention_mask[i].unsqueeze(0).to(self.device)
                    labels_for_loss = label_ids.clone()
                    labels_for_loss[label_mask == 0] = -100
                else:
                    labels_for_loss = label_ids

                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels_for_loss,
                )
                # 'outputs.loss' is mean cross-entropy over tokens for this label
                losses.append(float(outputs.loss.detach().cpu().item()))

        # Convert negative losses to probabilities
        neg_losses = torch.tensor([-l for l in losses], dtype=torch.float32)
        probs = torch.softmax(neg_losses, dim=0)
        return probs

    def __call__(self, image, labels):
        # Override to use joint image+text processing for BLIP
        if not self._initialized:
            self.load_model()
            self._initialized = True

        losses = []
        with torch.no_grad():
            for label in labels:
                inputs = self.processor(images=image, text=label, return_tensors="pt")
                pixel_values = inputs.get("pixel_values")
                input_ids = inputs.get("input_ids")
                attention_mask = inputs.get("attention_mask")

                if pixel_values is None or input_ids is None:
                    raise ValueError("BLIP processor did not return expected tensors")

                pixel_values = pixel_values.to(self.device)
                label_input_ids = input_ids.to(self.device)
                label_attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

                # Ignore padding tokens in the loss
                if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token_id is not None:
                    pad_id = self.processor.tokenizer.pad_token_id
                    labels_for_loss = label_input_ids.clone()
                    labels_for_loss[labels_for_loss == pad_id] = -100
                else:
                    labels_for_loss = label_input_ids

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=label_input_ids,
                    attention_mask=label_attention_mask,
                    labels=labels_for_loss,
                )
                loss_val = outputs.loss
                if loss_val is None:
                    raise ValueError("BLIP model did not return a loss; check inputs")
                losses.append(float(loss_val.detach().cpu().item()))

        neg_losses = torch.tensor([-l for l in losses], dtype=torch.float32)
        probs = torch.softmax(neg_losses, dim=0).tolist()
        return dict(zip(labels, probs))


class AlignAdapter(VisionModelAdapter):
    """ALIGN dual-encoder adapter using AlignProcessor + AlignModel.

    Uses the model forward pass to obtain logits_per_image directly,
    then applies softmax to obtain label probabilities.
    """

    def load_model(self):
        source_path = self.local_model_path if self.local_model_path else self.model_id
        source_type = "local directory" if self.local_model_path else "HuggingFace"
        print(f"Loading ALIGN model from {source_type}: {source_path}")

        self.processor = AlignProcessor.from_pretrained(
            source_path,
            local_files_only=bool(self.local_model_path)
        )
        self.model = AlignModel.from_pretrained(
            source_path,
            local_files_only=bool(self.local_model_path)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_image(self, image):
        inputs = self.processor(images=image, return_tensors='pt')
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def process_text(self, labels):
        inputs = self.processor(text=labels, return_tensors='pt', padding=True, truncation=True)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def compute_similarities(self, image_inputs, text_inputs):
        # Merge inputs and run model to get logits_per_image
        inputs = {}
        inputs.update(image_inputs)
        inputs.update(text_inputs)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # [batch=1, num_labels]
        probs = logits_per_image.softmax(dim=1).squeeze(0)
        return probs

# Legacy functions for backward compatibility
def get_model_components(model_name, local_model_path=None):
    """Legacy function - redirects to new adapter system."""
    adapter = get_vision_adapter(model_name, local_model_path)
    if not adapter._initialized:
        adapter.load_model()
        adapter._initialized = True
    
    # Return old-style components dict for compatibility
    return {
        'adapter': adapter,
        'device': adapter.device
    }

def process_image(image, components):
    """Legacy function - uses adapter if available."""
    if 'adapter' in components:
        return components['adapter'].process_image(image)
    
    # Fallback to old behavior
    device = components.get('device', torch.device('cpu'))
    if 'transform' in components:
        image_tensor = components['transform'](image).unsqueeze(0).to(device)
        return {'pixel_values': image_tensor}
    else:
        inputs = components['processor'](images=image, return_tensors='pt')
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

def process_text(labels, components):
    """Legacy function - uses adapter if available."""
    if 'adapter' in components:
        return components['adapter'].process_text(labels)
    
    # Fallback to old behavior
    device = components.get('device', torch.device('cpu'))
    if 'tokenizer' in components:
        inputs = components['tokenizer'](labels, return_tensors='pt', padding=True, truncation=True)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    else:
        inputs = components['processor'](text=labels, return_tensors='pt', padding=True)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

def crop_face(image, padding=50, side='largest'):
    """Detect and crop face from image.

    side: 'largest' | 'left' | 'right' | 'none'
    """
    if side == 'none':
        # Return original image without cropping
        return image

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = FACE_CASCADE or cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    if side == 'largest':
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        target = faces[0]
    elif side == 'left':
        faces = sorted(faces, key=lambda x: x[0])  # smallest x (leftmost)
        target = faces[0]
    elif side == 'right':
        faces = sorted(faces, key=lambda x: x[0], reverse=True)  # largest x (rightmost)
        target = faces[0]
    else:
        target = faces[0]

    (x, y, w, h) = target
    start_x, start_y = max(0, x - padding), max(0, y - padding)
    end_x, end_y = min(image_bgr.shape[1] - 1, x + w + padding), min(image_bgr.shape[0] - 1, y + h + padding)

    result = image_bgr[start_y:end_y, start_x:end_x]
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

def classify_image(image, labels, face, model_name="oai-base", local_model_path=None, model_architecture=None):
    """Classify image emotions using specified model.

    Args:
        image: Path to image file or URL
        labels: List of emotion labels to classify
        face: Face selection strategy ('largest', 'left', 'right', 'none')
        model_name: Name of HuggingFace model or predefined shorthand
        local_model_path: Optional path to local model directory
    """
    # Get appropriate adapter
    adapter = get_vision_adapter(model_name, local_model_path, model_architecture)
    
    # Load and prepare image
    if not image.startswith('http'):
        if not os.path.exists(image):
            raise ValueError("Image file does not exist at the specified path")
        pil_image = Image.open(image)
    else:
        try:
            pil_image = Image.open(requests.get(image, stream=True).raw)
        except:
            raise ValueError("Cannot retrieve image from URL")

    pil_image = pil_image.convert('RGB')
    
    # Crop face unless explicitly disabled
    processed_image = crop_face(pil_image, side=face)
    
    if processed_image is None:
        print('No face found in the image')
        return None
    
    # Use adapter for inference
    return adapter(processed_image, labels)

def classify_images_batch(images, labels, face='largest', model_name="oai-base", local_model_path=None, model_architecture=None):
    """Classify a batch of images efficiently by computing text embeddings once.

    Args:
        images: List of image file paths or URLs
        labels: List of labels
        face: 'largest' | 'left' | 'right' | 'none'
        model_name: Model alias or HF id
        local_model_path: Optional local model directory

    Returns:
        pandas.DataFrame with one row per image: columns ['image_id', *labels]
    """
    if not isinstance(images, (list, tuple)):
        raise ValueError("'images' must be a list of image paths/URLs")
    if not isinstance(labels, (list, tuple)) or len(labels) < 2:
        raise ValueError("'labels' must be a list with at least 2 items")

    # Get adapter and initialize if needed
    adapter = get_vision_adapter(model_name, local_model_path, model_architecture)
    if not adapter._initialized:
        adapter.load_model()
        adapter._initialized = True

    results = []
    device = adapter.device

    with torch.no_grad():
        # Compute text embeddings once for efficiency
        text_inputs = adapter.process_text(labels)
        text_embeds = adapter.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Get logit scale if available
        logit_scale = adapter.model.logit_scale.exp() if hasattr(adapter.model, 'logit_scale') else torch.tensor(1.0, device=device)

        for img in images:
            row = {label: np.nan for label in labels}
            row['image_id'] = os.path.basename(img) if isinstance(img, str) else str(img)
            
            try:
                # Load image
                if isinstance(img, str) and not img.startswith('http'):
                    if not os.path.exists(img):
                        results.append(row)
                        continue
                    pil_image = Image.open(img)
                else:
                    pil_image = Image.open(requests.get(img, stream=True).raw)
                
                pil_image = pil_image.convert('RGB')
                face_img = crop_face(pil_image, side=face)
                
                if face_img is None:
                    results.append(row)
                    continue
                
                # Process image and compute similarities
                image_inputs = adapter.process_image(face_img)
                image_embeds = adapter.model.get_image_features(**image_inputs)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                
                logits = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
                probs = logits.softmax(dim=1).squeeze(0).tolist()
                
                for lbl, p in zip(labels, probs):
                    row[lbl] = float(p)
                    
            except Exception:
                # Keep NaNs on failure
                pass
            finally:
                # Clean temporary tensors
                try:
                    del image_embeds
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            results.append(row)

        # Cleanup text embeddings
        try:
            del text_embeds
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ensure consistent column order
    df = pd.DataFrame(results)
    cols = ['image_id'] + list(labels)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    return df

# ============================================================================
# Model Registry and Factory Functions
# ============================================================================

# Adapter registry mapping model names to adapter classes
ADAPTER_REGISTRY = {
    "oai-base": CLIPAdapter,
    "oai-large": CLIPAdapter,
    "eva-8B": EVACLIPAdapter,
    "jina-v2": JinaCLIPAdapter,
    "blip-base": BLIPAdapter,
    "align-base": AlignAdapter,
}

# Model ID mapping (for backward compatibility)
MODEL_ID_MAP = {
    "oai-base": "openai/clip-vit-base-patch32",
    "oai-large": "openai/clip-vit-large-patch14",
    "eva-8B": "BAAI/EVA-CLIP-8B-448",
    "jina-v2": "jinaai/jina-clip-v2",
    "blip-base": "Salesforce/blip-image-captioning-base",
    "align-base": "kakaobrain/align-base",
}

# Global registry for loaded adapters (for caching)
_loaded_adapters = {}

def get_vision_adapter(model_name, local_model_path=None, architecture=None):
    """Factory function to get appropriate vision model adapter.
    
    Args:
        model_name: Model name/alias or HuggingFace model ID
        local_model_path: Optional local model path
        
    Returns:
        VisionModelAdapter: Appropriate adapter instance
    """
    # Create cache key
    cache_key = f"{model_name}_{local_model_path or 'remote'}"
    
    # Return cached adapter if available
    if cache_key in _loaded_adapters:
        return _loaded_adapters[cache_key]
    
    # Priority 1: explicit architecture hint from R registry
    adapter_class = None
    if architecture is not None:
        arch = str(architecture).lower()
        if arch.startswith("blip") or "blip" in arch:
            adapter_class = BLIPAdapter
            print(f"Using BLIP adapter (via registry) for: {model_name}")
        elif arch.startswith("align") or "align" in arch:
            adapter_class = AlignAdapter
            print(f"Using ALIGN adapter (via registry) for: {model_name}")
        elif arch.startswith("clip"):
            adapter_class = CLIPAdapter
            print(f"Using CLIP adapter (via registry) for: {model_name}")
        # else unknown; fall through

    # Priority 2: built-in alias mapping
    if adapter_class is None and model_name in ADAPTER_REGISTRY:
        adapter_class = ADAPTER_REGISTRY[model_name]
        model_id = MODEL_ID_MAP.get(model_name, model_name)
    else:
        model_id = model_name

    # Priority 3: heuristic by model id/name
    if adapter_class is None:
        lower_name = model_name.lower()
        if "blip" in lower_name:
            adapter_class = BLIPAdapter
            print(f"Using BLIP adapter for model: {model_name}")
        elif "align" in lower_name:
            adapter_class = AlignAdapter
            print(f"Using ALIGN adapter for model: {model_name}")
        elif "jina" in lower_name:
            adapter_class = JinaCLIPAdapter
            print(f"Using Jina CLIP adapter for model: {model_name}")
        elif "eva" in lower_name:
            adapter_class = EVACLIPAdapter
            print(f"Using EVA CLIP adapter for model: {model_name}")
        else:
            adapter_class = CLIPAdapter
            print(f"Using standard CLIP adapter for model: {model_name}")
    
    # Create and cache adapter
    adapter = adapter_class(model_id, local_model_path)
    _loaded_adapters[cache_key] = adapter
    
    return adapter

def register_custom_adapter(model_name, adapter_class, model_id=None):
    """Register a custom adapter for a model.
    
    Args:
        model_name: Name/alias for the model
        adapter_class: VisionModelAdapter subclass
        model_id: Optional HuggingFace model ID (defaults to model_name)
    """
    if not issubclass(adapter_class, VisionModelAdapter):
        raise ValueError("adapter_class must be a subclass of VisionModelAdapter")
    
    ADAPTER_REGISTRY[model_name] = adapter_class
    MODEL_ID_MAP[model_name] = model_id or model_name
    print(f"Registered custom adapter '{model_name}' -> {adapter_class.__name__}")

# For backward compatibility - maintain old dictionaries
available_models = MODEL_ID_MAP.copy()
model_dict = MODEL_ID_MAP.copy()
