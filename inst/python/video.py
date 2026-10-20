import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import cv2
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from image import classify_image
import te_faces
import torch.nn.functional as F
import time


def yt_analyze(url, nframes, labels, side='largest', start=0, end=-1, uniform=False,
               ff=10, frame_dir='temp/', video_name='temp', model_name='oai-base', local_model_path=None,
               model_architecture=None, face_detector='haar', youtube=False):
  start_time = time.time()
  detected_emotions = []
  if youtube:  # decided in R by .te_is_youtube_url()
    video_path = te_faces.download_youtube(url, frame_dir, video_name)
  else:
    video_path = os.path.join(url)
  # Only frames that were read are scored, so a frame that cannot be read no
  # longer shifts the frame numbering
  frames = te_faces.extract_frames(video_path, nframes, start=start, end=end,
                                   uniform=uniform, ff=ff, frame_dir=frame_dir,
                                   video_name=video_name)

  for frame in frames:
    emotions = classify_image(
      frame["path"],
      labels,
      face=side,
      model_name=model_name,
      local_model_path=local_model_path,
      model_architecture=model_architecture,
      face_detector=face_detector,
    )
    if emotions:
      # Add the dictionary directly to our list
      detected_emotions.append(emotions)
    else:
      # Create a dictionary with NaN values for each emotion
      detected_emotions.append({label: np.nan for label in labels})

  # Convert list of dictionaries to DataFrame
  df = pd.DataFrame(detected_emotions, columns=list(labels))
  end_time = time.time()
  print(f"Done! Execution time: {end_time - start_time} seconds")
  return df
