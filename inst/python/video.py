import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import cv2 
import numpy as np
import pandas as pd
from pytube import YouTube
import face_recognition
from transformers import AutoProcessor, AutoModel
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import time
import shutil


def yt_analyze(url, nframes, labels, probability=True, side='largest', cut = 'no', start = 0, end=60, uniform = False, ff = 10, save_video = False, save_frames = False, frame_dir = 'temp/', video_name = 'temp'):
  text_embeds_openai = get_text_embeds(labels)
  start_time = time.time()
  temp_dir =  frame_dir
  if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
  yt = YouTube(url)
  k = 0
  video = None
  detected_emotions = []
  while k < 3: # make three attempts to retrieve YouTube video stream
    try:
      video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
      break
    except:
      print("Error occurred while getting video stream. Retrying ... ")
      k += 1
  if k==3:
    raise ValueError("Failed to get video stream after 3 attempts.")
  else:
    video_path = os.path.join(temp_dir, video_name + ".mp4")
    print(f"Downloading video to {video_path}")
    k = 0
    while k < 3:
      try:
        video.download(output_path=frame_dir, filename="%s.mp4" %video_name)
        break
      except:
        print("Error occurred while downloading video. Retrying...")
        k += 1
    if k==3:
      raise ValueError("Failed to download video after 3 attempts.")
    else:
      cap = cv2.VideoCapture(video_path)
      if cut == 'no':
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if uniform:
         frame_interval = math.ceil(num_frames / nframes)
        else:
          frame_interval = ff
        if ff*nframes > num_frames:
          raise ValueError("Video is too short for requested number of frames.")
        counter = 0
        print(f"Extracting {nframes} frames from {num_frames} total frames")
        for i in range(nframes):
            frame_index = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
              print(f"Error reading frame {i}")
              continue
            image_path = os.path.join(frame_dir, f"{video_name}-frame-{i}.jpg")
            cv2.imwrite(image_path, frame)
            counter += 1
            ################################## CONTINUE HERE
      else:
        start_frame = int(start * cap.get(cv2.CAP_PROP_FPS))
        end_frame = int(end * cap.get(cv2.CAP_PROP_FPS))
        print("Cutting video to designated time interval...")
        num_frames = end_frame - start_frame
        if num_frames < nframes:
          frame_interval = 1
        else:
          if uniform:
            frame_interval = math.ceil(num_frames / nframes)
          else:
            frame_interval = ff
        print(f"Extracting {nframes} frames from {num_frames} total frames")
        counter = 0
        for i in range(min(nframes, num_frames)):
            frame_index = start_frame + i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
              print(f"Error reading frame {i}")
              continue
            image_path = os.path.join(temp_dir, f"frame_{i}.jpg")
            cv2.imwrite(image_path, frame)
            counter += 1
      print(f"Total number of saved frames: {counter}")
      n = counter
      
      for i in range(counter):
        image_path = os.path.join(temp_dir, f"frame_{i}.jpg")
        image = Image.open(image_path)
        image = crop_face(image)
        if not(image is None):
          emotions = classify_openai(image,labels, text_embeds_openai)
        else:
          detected_emotions.append([np.nan]*len(labels))
        if emotions:
          detected_emotions.append(list(emotions.values()))
        else:
          detected_emotions.append([np.nan]*len(labels))
      df = pd.DataFrame(detected_emotions)
      df.columns = labels
      if not save_frames:
        # remove all pngs files from frame dir
        for file in os.listdir(temp_dir):
          if file.endswith(".png"):
            os.remove(os.path.join(temp_dir, file))
      if not save_video:
        os.remove(video_path)
      end_time = time.time()
      print(f"Execution time: {end_time - start_time} seconds")
      return df 
