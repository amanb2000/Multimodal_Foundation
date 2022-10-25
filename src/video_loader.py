""" 
Functions for loading the video dataset. Patching & positional encoding is 
in `preprocessing.py`. 

Prototyped in `/sketch/02_Video_Loading.ipynb`.
"""

## Import box 
import os 
import sys 
import random
import pathlib
import itertools
import collections
import random
import pdb

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pathlib 
from tqdm import tqdm
import cv2



def get_videoset(data_folder, num_videos, nframes=101, random_select=True, 
		batch_size=1, output_size=(256,256), hard_crop=True): 
	""" Primary method for obtaining the video Dataset.

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
	## 1: Get the list of paths to load.
	assert os.path.exists(data_folder), "Data folder does not exist!"
	mp4list = os.listdir(data_folder) 
	mp4list = [l for l in mp4list if l.endswith("mp4")]

	if random_select: 
		random.shuffle(mp4list)
	mp4list = mp4list[:num_videos]
	pathlist = [os.path.join(data_folder, i) for i in mp4list] # final pathlist

	## 2: Load the paths, convert to tensors/dataset. 
	# video_tensor_list = [get_single_video_tensor(i, nframes, output_size=output_size, hard_crop=hard_crop) for i in pathlist]
	video_tensor_list = []

	for i in tqdm(pathlist):
		new_tensor = get_single_video_tensor(i, nframes, output_size=output_size, hard_crop=hard_crop)
		video_tensor_list.append(new_tensor)

	VideoSet = tf.data.Dataset.from_tensor_slices(video_tensor_list)
	VideoSet = VideoSet.batch(batch_size)

	## 3: Return
	return VideoSet

def get_single_video_tensor(video_path, n_frames, output_size = (256,256), hard_crop=True):
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

	assert os.path.exists(video_path), "Video path does not exist!"
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
	if len(result) == 0:
		print(f"Unable to read in any frames in {video_path}. Check that the video path is valid.")
		return None

	result = np.array(result)[..., [2, 1, 0]]

	return result


if __name__ == "__main__":
	## Testing out functions 
	VideoSet = get_videoset("../datasets/downloads", 10)
	print("\nSuccessfully obtained VideoSet: ", VideoSet)