import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import face_recognition
from transformers import AutoProcessor, AutoModel
import torch 
import requests
from PIL import Image

def crop_face(image, padding=50, side='largest'):
  # Convert the image to a numpy array
  image_array = np.array(image)
  # Find all the faces in the image
  face_locations = face_recognition.face_locations(image_array)
  # If no faces are found, return the original image
  if len(face_locations) == 0:
    return None
  # Find the face to crop
  if side == 'largest':
    face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
  elif side == 'left':
    face_location = min(face_locations, key=lambda loc: loc[3])
  elif side == 'right':
    face_location = max(face_locations, key=lambda loc: loc[1])
  else:
    raise ValueError("Invalid value for 'side' argument")
  # Add padding to the face bounding box
  top, right, bottom, left = face_location
  top = max(0, top - padding)
  right = min(image_array.shape[1], right + padding)
  bottom = min(image_array.shape[0], bottom + padding)
  left = max(0, left - padding)
  # Crop the image to the selected face
  cropped_image = Image.fromarray(image_array[top:bottom, left:right])
  # Return the cropped imagef
  return cropped_image

def classify_openai(image,labels,text_embeds_openai):
  with torch.no_grad():
    # if not url
    if not image.startswith('http'):   
      image = os.path.join(image)
      image = Image.open(image)
    else :
      # try to get image from url
      try :
        image = requests.get(image, stream=True).raw
      except :
        raise ValueError("Cannot retrieve image from URL")
    image = image.convert('RGB')
    image = crop_face(image)
    image_inputs = processor_openai(images=image, return_tensors='pt')
    image_embeds = model_openai.get_image_features(**image_inputs)
    image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)
    logits_per_image = torch.matmul(image_embeds, text_embeds_openai.t()) * model_openai.logit_scale.exp()
    probs = logits_per_image.softmax(dim=1).squeeze(0).tolist()
    del logits_per_image
    del image_inputs
    del image_embeds
    torch.cuda.empty_cache()
    return dict(zip(labels, probs))
  
def get_text_embeds(labels):
  global processor_openai
  processor_openai = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
  global model_openai
  model_openai = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
  text_inputs_openai = processor_openai(text=labels, return_tensors='pt', padding=True)
  text_embeds_openai = model_openai.get_text_features(**text_inputs_openai)
  text_embeds_openai /= text_embeds_openai.norm(p=2, dim=-1, keepdim=True)
  return text_embeds_openai
