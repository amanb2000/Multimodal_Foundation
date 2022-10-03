""" 
Functions for loading the video dataset. 

Prototyped in `/sketch/02_Video_Loading.ipynb`.
"""

## Import box 
import os 
import sys 
import random
import pathlib
import itertools
import collections

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pathlib 
from tqdm.notebook import tqdm
import cv2
# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed


def get_videoset(data_folder, num_videos, nframes=101, random_select=True, 
		batch_size=1, output_size=(256,256)): 
	""" Primary method for obtaining the video dataset. 

	Arguments: 
		`data_folder`: 		Path to the enclosing folder with all the .mp4 files.
		`num_videos`: 		Number of videos to load. 

	Keyword arguments:
		`nframes`: 			Number of frames (starting from beginning) to load. 
							-1 for loading all frames.
		`random_select`: 	Should we randomize the selection of `num_videos`? 
		`batch_size`: 		Batch size for the Dataset object returned to user. 
		`output_size`: 		Tuple of (height, width) of the videos (cropped).

	Returns: 
		`VideoSet`: 		Tensorflow dataset of the requested videos. Each 
							element has dimensions 
							[batch_size, nframes, height, width, channels].
	"""
	

def get_numpy_video_tensor(video_path, n_frames, output_size = (256,256), hard_crop=True):
	""" Creates frames from each video file present for each category.

	Args:
		video_path: File path to the video.
		n_frames: Number of frames to be created per video file.
		output_size: Pixel size of the output frame image.
		hard_crop: Do we cut off edges along one dimension to make the video fit? Else we add black bars. 

	Return:
		An NumPy array of frames in the shape of (n_frames, height, width, channels).
	"""
	count = 0

	result = [] # List of frames, converted to a numpy tensor at the end.

	src = cv2.VideoCapture(str(video_path))
	video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

	start = 0 # starting frame

	if n_frames == -1: 
		n_frames = int(video_length)

	src.set(cv2.CAP_PROP_POS_FRAMES, start)
	for i in range(n_frames):
		ret, frame = src.read()
		if ret:
			frame = tf.image.convert_image_dtype(frame, tf.float32) 
			frame = frame[tf.newaxis, ...]


			b,h,w,c = frame.shape # true height, width, channels
			height_mult = h/output_size[0] # true/desired
			width_mult = w/output_size[1]

			if height_mult < width_mult: 
				new_height = output_size[0] 
				new_width = int(w / height_mult)
			else:
				new_width = output_size[1] 
				new_height = int(h / width_mult)
			
			frame = tf.image.resize(frame, (new_height, new_width))

			if hard_crop:
				frame = tf.image.resize_with_crop_or_pad(frame, *output_size)
			else:
				frame = tf.image.resize_with_pad(frame, *output_size)

			frame = tf.squeeze(frame)
			result.append(frame)
		else:
			break # no more frames!
	
	src.release()
	# Ensure that the color scheme is not inverted
	assert len(result) != 0, "Unable to read in any frames. Check that the video path is valid."

	result = np.array(result)[..., [2, 1, 0]]

	return result