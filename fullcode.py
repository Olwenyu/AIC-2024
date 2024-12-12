import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from glob import glob

import requests
import tkinter as tk
import webbrowser

from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel

import numpy as np
from glob import glob
import json
import os

# from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import models
import clip

import streamlit as st
from PIL import Image
import pandas as pd
import csv


device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)


def calculate(inputs,v):
    if inputs == "":
        return None
    prompt = inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    features_path = ("D:\\AIC\\filenpy")

    top_scores = []
    text_embedding = model.get_text_features(**inputs)
    text_embedding = text_embedding.cpu().detach().numpy()

    if(v == ""):
        for feature in sorted(os.listdir(features_path)):
            frame_features = np.load(os.path.join(features_path, feature))


            frame_features = torch.tensor(frame_features)
            transpose = frame_features.T
            frame_features = transpose / np.linalg.norm(frame_features, axis=1)
            scores = np.dot(text_embedding, frame_features)

            top = 100
            idx = np.argsort(-scores[0])[:top]

            base_name = os.path.basename(feature)[:8]

            # Append the top scores with file names and frame indices
            for frame_idx in idx:
                top_scores.append((base_name, frame_idx, scores[0][frame_idx]))
    else:
        frame_features = np.load(f'{features_path}\\{v}.npy')


        frame_features = torch.tensor(frame_features)
        transpose = frame_features.T
        frame_features = transpose / np.linalg.norm(frame_features, axis=1)
        scores = np.dot(text_embedding, frame_features)

        top = 100
        idx = np.argsort(-scores[0])[:top]

        base_name = os.path.basename(v)[:8]

        # Append the top scores with file names and frame indices
        for frame_idx in idx:
            top_scores.append((base_name, frame_idx, scores[0][frame_idx]))

    top_scores.sort(key=lambda x: -x[2])
    return top_scores

#MAP
def find_frame_index(csv_file, video_name, frame_number):
    df = pd.read_csv(csv_file)
    value = df.loc[frame_number, 'frame_index']
    return value


def shorten_url(url):
    try:
        response = requests.get(f'http://tinyurl.com/api-create.php?url={url}')
        response.raise_for_status()  # Ensure we notice bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error shortening URL: {e}")
        return url
    
def open_link(url):
    webbrowser.open_new(url)
def youtube_link(video_name, frame_index):
    try:
        with open(f'D:\\AIC\\media-info-b1\\media-info\\{video_name}.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

    # Extract watch_url
        video_url = data["watch_url"]

        # Calculate the time in seconds
        if (video_name != 'L_V'):
            frame_rate = 25 
        else:
            frame_rate = 30
            
        frame_index = int(frame_index) # convert the input to an integer
        time_in_seconds = frame_index / frame_rate

        # round to 3 decimal places for millisecond precision
        time_in_seconds = round(time_in_seconds, 3)

        # Append the time parameter to the video URL
        youtube_link = f"{video_url}&t={time_in_seconds}s"
        youtube_link = shorten_url(youtube_link)
        return youtube_link
    except FileNotFoundError:
        return f"Video Name: {video_name}, Error: Json file not found"

# PATH
keyframes_folders_all = "D:\\AIC\\We\\Keyframes"  # chỗ chứa Keyframes_L01, Keyframes_L02,...

# PROCESS

st.set_page_config(layout="wide")

v = st.text_input("Video name")
inputs = st.text_input("Prompt")
output = calculate(inputs,v)
rows = []

if output == None:
    st.write("NOTHING TO SHOW")
else:
    num_columns = 5  # Number of columns in the grid
    columns = st.columns(num_columns)

    with open("D:\\AIC\\submission\\output.csv", mode='w', newline='') as file:
        writer = csv.writer(file)

        for i, (file_name, frame_idx, score) in enumerate(output[:300]):
            frame_index = find_frame_index(f"D:\\AIC\\We\\mapkeyframes\\{file_name}.csv",file_name,frame_idx)
            image_path = f"{keyframes_folders_all}\\Keyframes_{file_name[:3]}\\keyframes\\{file_name}\\{frame_idx:04}.webp"
            if (os.path.exists(image_path)):
                name_youtube = youtube_link(file_name, frame_index)
                writer.writerow([file_name,frame_index])
                image = Image.open(image_path)
                with columns[i % num_columns]:
                    st.image(
                        image,
                        caption=f"Top={i+1}, File={file_name}, Key frame={frame_idx}, Frame idx={frame_index}, Score={score:.4f}",
                    )
                    st.write(f"[Video Link]({name_youtube})")
        
