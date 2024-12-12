import json
import streamlit as st
from streamlit import components
import pandas as pd    

video_name = st.text_input("Enter the video name")
frame_index = st.number_input("Enter the frame number")

# df = pd.read_csv(f"D:\\AIC\\We\\mapkeyframes\\{video_name}.csv")
# frame_index = df.loc[frame_number, 'frame_index']


# Add a button to submit the input
st.write(f'Video Name: {video_name}')
st.write(f'Frame index: {frame_index}')


# Load the JSON data
#print(video_name)
with open(f'D:\\AIC\\media-info-b1\\media-info\\{video_name}.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract watch_url
video_url = data["watch_url"]

# Calculate the time in seconds
# D:\AIC\map-keyframes-b1\map-keyframes\L06_V003.csv 30.0
# D:\AIC\map-keyframes-b1\map-keyframes\L09_V009.csv 30.0
if (video_name == 'L06_V003' or video_name == 'L09_V009'):
    frame_rate = 30 
else:
    frame_rate = 25
frame_index = int(frame_index) # convert the input to an integer
time_in_seconds = frame_index / frame_rate

# round to 3 decimal places for millisecond precision
time_in_seconds = round(time_in_seconds, 10)

# Append the time parameter to the video URL
youtube_link = f"{video_url}&t={time_in_seconds}s"

st.write(f"[Video Link]({youtube_link})")
st.video(youtube_link)