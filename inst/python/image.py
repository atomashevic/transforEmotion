import os
import urllib.request
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from transformers import AutoProcessor, AutoModel
import torch 
import requests
import cv2
from PIL import Image

def crop_face(image, padding=50, side='largest'):
    # image 
    # from PIL image to cv2 image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Convert the image to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade xml file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are found, return the original image
    if len(faces) == 0:
        return None

    # If 'side' is 'largest', find the largest face
    if side == 'largest':
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    # Get the bounding box of the first face
    (x, y, w, h) = faces[0]

    # Crop the face with padding
    start_x, start_y = max(0, x - padding), max(0, y - padding)
    end_x, end_y = min(image.shape[1] - 1, x + w + padding), min(image.shape[0] - 1, y + h + padding)

    result = image[start_y:end_y, start_x:end_x]
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(result)
    return result

def classify_image(image, labels, face, model_name='openai/clip-vit-large-patch14'):
  text_embeds = get_text_embeds(labels, model_name)
  with torch.no_grad():
    # if not url
    if not image.startswith('http'):   
      # check  if image exists
      if not os.path.exists(image):
        raise ValueError("Image file does not exist at the specified path")
      else:
        image = os.path.join(image)
        image = Image.open(image)
    else :
      # try to get image from url
      try :
        image = requests.get(image, stream=True).raw
        image = Image.open(image)
      except :
        raise ValueError("Cannot retrieve image from URL")
    image = image.convert('RGB')
    image = crop_face(image, side=face)
    if image != None:
      model = model_dict[model_name]['model']
      processor = model_dict[model_name]['processor']
      image_inputs = processor(images=image, return_tensors='pt')
      image_embeds = model.get_image_features(**image_inputs)
      image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)
      logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * model.logit_scale.exp()
      probs = logits_per_image.softmax(dim=1).squeeze(0).tolist()
      del logits_per_image
      del image_inputs
      del image_embeds
      torch.cuda.empty_cache()
      return dict(zip(labels, probs))
    else :
      print('No face found in the image')
      return None
  
model_dict = {}

def get_text_embeds(labels, model_name):
  global model_dict
  if model_name not in model_dict:
    print(f"Loading model {model_name} from HuggingFace...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model_dict[model_name] = {'processor': processor, 'model': model}
  else:
    processor = model_dict[model_name]['processor']
    model = model_dict[model_name]['model']
  
  text_inputs = processor(text=labels, return_tensors='pt', padding=True)
  text_embeds = model.get_text_features(**text_inputs)
  text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)
  return text_embeds
