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

warnings.filterwarnings("ignore") 
logging.set_verbosity_error()

def get_model_components(model_name, local_model_path=None):
    """Initialize or retrieve model components from cache.
    
    Args:
        model_name: Name of the model to load from HuggingFace
        local_model_path: Optional path to local model directory
    """
    model_path = model_dict.get(model_name, model_name)
    cache_key = local_model_path if local_model_path else model_path
    
    if cache_key not in model_dict:
        # Determine source path (HuggingFace or local)
        source_path = local_model_path if local_model_path else model_path
        source_type = "local directory" if local_model_path else "HuggingFace"
        print(f"Loading model from {source_type}: {source_path}")
        
        if model_name == "jina-v2":
            model = AutoModel.from_pretrained(source_path, trust_remote_code=True, local_files_only=bool(local_model_path))
            # For tokenizer, always use the standard one unless explicitly provided locally
            tokenizer_path = local_model_path if local_model_path else "openai/clip-vit-base-patch32"
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=bool(local_model_path))
            transform = T.Compose([
                T.Resize(512, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(512),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                          (0.26862954, 0.26130258, 0.27577711))
            ])
            model_dict[cache_key] = {
                'model': model,
                'tokenizer': tokenizer,
                'transform': transform
            }
        elif model_name == "eva-8B":
            print("\n" + "="*80)
            print("Loading EVA-CLIP-8B model (CPU-compatible mode)")
            print("NOTE: This model is very large (8 billion parameters).")
            print("Loading with low_cpu_mem_usage=True to reduce memory requirements.")
            print("This may take several minutes and requires at least 16GB of RAM.")
            print("If you experience memory issues, consider using a smaller model.")
            print("="*80 + "\n")
            
            try:
                # CPU-compatible loading with memory optimization
                model = CLIPModel.from_pretrained(
                    source_path,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,  # Use half precision to reduce memory
                    local_files_only=bool(local_model_path)
                )
                print("Successfully loaded EVA model with memory optimization")
            except Exception as e:
                # First error - try with safetensors
                print(f"Initial loading failed: {str(e)}")
                print("Trying alternative loading method...")
                
                try:
                    model = CLIPModel.from_pretrained(
                        source_path,
                        ignore_mismatched_sizes=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        local_files_only=bool(local_model_path),
                        use_safetensors=True
                    )
                    print("Successfully loaded EVA model with safetensors")
                except Exception as e2:
                    # Second failure - notify user that model is too large
                    print("\n" + "="*80)
                    print(f"ERROR: Could not load EVA model: {str(e2)}")
                    print("This model is too large for your current hardware configuration.")
                    print("Please consider using a smaller model like 'oai-large' instead.")
                    print("="*80)
                    raise ImportError("EVA model is too large for current hardware")
            transform = T.Compose([
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                          (0.26862954, 0.26130258, 0.27577711))
            ])
            # For tokenizer, always use the standard one unless explicitly provided locally
            tokenizer_path = local_model_path if local_model_path else "openai/clip-vit-base-patch32"
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=bool(local_model_path))
            model_dict[cache_key] = {
                'model': model,
                'tokenizer': tokenizer,
                'transform': transform
            }
        else:
            processor = CLIPProcessor.from_pretrained(source_path, local_files_only=bool(local_model_path))
            model = CLIPModel.from_pretrained(source_path, ignore_mismatched_sizes=True, local_files_only=bool(local_model_path))
            model_dict[cache_key] = {
                'model': model,
                'processor': processor
            }
    
    return model_dict[cache_key]

def process_image(image, components):
    """Process image according to model requirements."""
    if 'transform' in components:
        image_tensor = components['transform'](image).unsqueeze(0)
        return {'pixel_values': image_tensor}
    else:
        return components['processor'](images=image, return_tensors='pt')

def process_text(labels, components):
    """Process text according to model requirements."""
    if 'tokenizer' in components:
        return components['tokenizer'](labels, return_tensors='pt', padding=True, truncation=True)
    else:
        return components['processor'](text=labels, return_tensors='pt', padding=True)

def crop_face(image, padding=50, side='largest'):
    """Detect and crop face from image."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    if side == 'largest':
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    (x, y, w, h) = faces[0]
    start_x, start_y = max(0, x - padding), max(0, y - padding)
    end_x, end_y = min(image.shape[1] - 1, x + w + padding), min(image.shape[0] - 1, y + h + padding)

    result = image[start_y:end_y, start_x:end_x]
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

def classify_image(image, labels, face, model_name="oai-base", local_model_path=None):
    """Classify image emotions using specified model.
    
    Args:
        image: Path to image file or URL
        labels: List of emotion labels to classify
        face: Face selection strategy ('largest', 'left', 'right')
        model_name: Name of HuggingFace model or predefined shorthand
        local_model_path: Optional path to local model directory
    """
    components = get_model_components(model_name, local_model_path)
    
    with torch.no_grad():
        # Load and prepare image
        if not image.startswith('http'):
            if not os.path.exists(image):
                raise ValueError("Image file does not exist at the specified path")
            image = Image.open(image)
        else:
            try:
                image = Image.open(requests.get(image, stream=True).raw)
            except:
                raise ValueError("Cannot retrieve image from URL")
        
        image = image.convert('RGB')
        image = crop_face(image, side=face)
        
        if image is None:
            print('No face found in the image')
            return None
            
        # Process inputs
        image_inputs = process_image(image, components)
        text_inputs = process_text(labels, components)
        
        # Generate embeddings
        model = components['model']
        image_embeds = model.get_image_features(**image_inputs)
        text_embeds = model.get_text_features(**text_inputs)
        
        # Normalize embeddings
        image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate similarity and probabilities
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * model.logit_scale.exp()
        probs = logits_per_image.softmax(dim=1).squeeze(0).tolist()
        
        # Clean up
        del logits_per_image, image_embeds, text_embeds
        torch.cuda.empty_cache()
        
        return dict(zip(labels, probs))

available_models = {
    "oai-base": "openai/clip-vit-base-patch32",
    "oai-large": "openai/clip-vit-large-patch14",
    "eva-8B": "BAAI/EVA-CLIP-8B-448",
    "jina-v2": "jinaai/jina-clip-v2"
}

model_dict = {
    "oai-base": "openai/clip-vit-base-patch32",
    "oai-large": "openai/clip-vit-large-patch14",
    "eva-8B": "BAAI/EVA-CLIP-8B-448",
    "jina-v2": "jinaai/jina-clip-v2"
}
