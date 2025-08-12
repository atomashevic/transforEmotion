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

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Load face detector once to avoid per-image overhead
try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    FACE_CASCADE = None

def get_model_components(model_name, local_model_path=None):
    """Initialize or retrieve model components from cache.

    Args:
        model_name: Name of the model to load from HuggingFace
        local_model_path: Optional path to local model directory
    """
    model_path = model_dict.get(model_name, model_name)
    cache_key = local_model_path if local_model_path else model_path
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cache_key not in model_dict:
        # Determine source path (HuggingFace or local)
        source_path = local_model_path if local_model_path else model_path
        source_type = "local directory" if local_model_path else "HuggingFace"
        print(f"Loading model from {source_type}: {source_path}")

        if model_name == "jina-v2":
            model = AutoModel.from_pretrained(source_path, trust_remote_code=True, local_files_only=bool(local_model_path))
            model = model.to(device)
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
                'transform': transform,
                'device': device
            }
        elif model_name == "eva-8B":
            try:
                print("Attempting to load EVA-CLIP-8B model with 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",  # normalized float 4
                    bnb_4bit_use_double_quant=True,  # use nested quantization for more memory efficiency
                )
                model = CLIPModel.from_pretrained(
                    model_path,
                    ignore_mismatched_sizes=True,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    local_files_only=bool(local_model_path)
                )
                model = model.to(device)
                print("Successfully loaded EVA-CLIP-8B model")
            except Exception as e:
                print(f"Quantized model loading failed: {str(e)}")
                print("Falling back to standard loading method...")
                try:
                    model = CLIPModel.from_pretrained(
                        model_path,
                        ignore_mismatched_sizes=True,
                        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                        local_files_only=bool(local_model_path)
                    )
                    model = model.to(device)
                    print("Successfully loaded EVA-CLIP-8B model without quantization.")
                except Exception as e2:
                    print(f"Standard model loading failed: {str(e2)}")
                    raise ImportError("EVA-CLIP-8B model could not be loaded. Please check your setup.")

            # Define the image transformation pipeline
            transform = T.Compose([
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))
            ])

            # Load the tokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            # Store the model, tokenizer, and transform in the model dictionary
            model_dict[model_path] = {
                'model': model,
                'tokenizer': tokenizer,
                'transform': transform,
                'device': device
            }
        else:
            processor = CLIPProcessor.from_pretrained(source_path, local_files_only=bool(local_model_path))
            model = CLIPModel.from_pretrained(source_path, ignore_mismatched_sizes=True, local_files_only=bool(local_model_path))
            model = model.to(device)
            model_dict[cache_key] = {
                'model': model,
                'processor': processor,
                'device': device
            }

    return model_dict[cache_key]

def process_image(image, components):
    """Process image according to model requirements."""
    device = components.get('device', torch.device('cpu'))
    if 'transform' in components:
        image_tensor = components['transform'](image).unsqueeze(0).to(device)
        return {'pixel_values': image_tensor}
    else:
        inputs = components['processor'](images=image, return_tensors='pt')
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

def process_text(labels, components):
    """Process text according to model requirements."""
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
        # Crop face unless explicitly disabled
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dict(zip(labels, probs))

def classify_images_batch(images, labels, face='largest', model_name="oai-base", local_model_path=None):
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

    components = get_model_components(model_name, local_model_path)
    model = components['model']

    results = []
    device = components.get('device', torch.device('cpu'))

    with torch.no_grad():
        # Compute text embeddings once
        text_inputs = process_text(labels, components)
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else torch.tensor(1.0, device=device)

        for img in images:
            row = {label: np.nan for label in labels}
            row['image_id'] = os.path.basename(img) if isinstance(img, str) else str(img)
            try:
                if isinstance(img, str) and not img.startswith('http'):
                    if not os.path.exists(img):
                        # leave NaNs
                        results.append(row)
                        continue
                    pil = Image.open(img)
                else:
                    pil = Image.open(requests.get(img, stream=True).raw)
                pil = pil.convert('RGB')
                face_img = crop_face(pil, side=face)
                if face_img is None:
                    results.append(row)
                    continue
                image_inputs = process_image(face_img, components)
                image_embeds = model.get_image_features(**image_inputs)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                logits = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
                probs = logits.softmax(dim=1).squeeze(0).tolist()
                for lbl, p in zip(labels, probs):
                    row[lbl] = float(p)
            except Exception:
                # keep NaNs on failure
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

        # Optional cleanup
        try:
            del text_embeds
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ensure consistent column order: image_id then label columns
    df = pd.DataFrame(results)
    cols = ['image_id'] + [c for c in labels]
    # Add any missing columns (if results was empty)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    return df

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
