""" 
Here I prototype and benchmark a dataset loading system that randomly selects videos 
from the source video folder to load in batches. 

I will use python's `multiprocessing` module to achieve large scale parallelism. 
"""
import os 
# import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
from multiprocessing import Pool
import random
import time
import cv2

import numpy as np




print("\n\n\n")

DATA_FOLDER = "../datasets/downloads"
mp4_list = os.listdir(DATA_FOLDER)
print("MP4 LIST: ", mp4_list[:10])



def get_multi_video_tensor(video_path, n_frames, output_size=(256,256), pool_size=3): 
	""" Applies multiprocessing to obtain a single video tensor. 
	"""

	print("POOL SIZE IN MULTI VIDEO RETREIVAL: ", pool_size)
	assert os.path.exists(video_path), "Video path does not exist!"
	src = cv2.VideoCapture(str(video_path))
	video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

	max_start = video_length-n_frames-1
	if max_start < 0: 
		print(f"Video {video_path} not long enough for {n_frames} frame clip!")
		return None
	
	try:
		start = random.randint(0, max_start) # starting frame
	except:
		return None

	if n_frames == -1: 
		n_frames = int(video_length)

	start_inc = n_frames/pool_size
	call_args = [ [video_path, int(start_inc), int(start+i*start_inc), output_size] for i in range(pool_size)]

	for x in call_args:
		print(x)

	with Pool(pool_size) as p:
		video_segments = p.starmap(get_single_video_tensor, call_args)


	video_total = np.concatenate(video_segments, axis=1)

	return video_total



def get_single_video_tensor(video_path, n_frames, start_frame, output_size):
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

	src.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	for i in range(n_frames):
		ret, frame = src.read()
		if ret:
			frame_size = frame.shape[:2]

			scale_factor = max(output_size[0]/frame_size[0], output_size[1]/frame_size[1])

			resized = cv2.resize(frame, [round(frame_size[1]*scale_factor), round(frame_size[0]*scale_factor)] )
			new_h, new_w, _ = resized.shape

			crop_height_total = new_h - output_size[0] 
			crop_width_total = new_w - output_size[1] 

			start_height = crop_height_total // 2
			start_width = crop_width_total // 2

			end_height = start_height + output_size[0]
			end_width = start_width + output_size[1]


			cropped = resized[start_height:end_height, start_width:end_width, :]
			frame = np.expand_dims(cropped, 0)

			result.append(frame)
		else:
			break # no more frames!
	
	src.release()
	# Ensure that the color scheme is not inverted
	if len(result) == 0:
		print(f"Unable to read in any frames in {video_path}. Check that the video path is valid.")
		return None

	# pdb.set_trace()
	result = np.concatenate(result)
	result = np.expand_dims(result,0)

	return result




def get_random_video(video_list, prefix="", depth=0, output_size=(120,180), 
		num_frames=100, subprocess_threads=3): 
	""" Selects a random video name from `video_list`, adds `prefix`, thenn 
	loads the video from that path. If that fails, we call this function again. 
	"""

	assert depth < 100, "Hit maximum recursion depth!"

	fname = random.choice(video_list)
	fpath = os.path.join(prefix, fname)
	retval = get_multi_video_tensor(fpath, num_frames, output_size=output_size, pool_size=subprocess_threads)
	if retval is None:
		print(f"`get_random_video` failed at depthh {depth}. Trying again...")
		return get_random_video(video_list, prefix=prefix, depth=depth+1, output_size=output_size, num_frames=num_frames)
	return retval

def print_args(video_list, prefix, depth, output_size, num_frames, subprocess_threads):
	return get_random_video(video_list, prefix=prefix, depth=depth, 
			output_size=output_size, num_frames=num_frames, subprocess_threads=subprocess_threads)

def get_random_batch(num_videos, video_list, prefix="", output_size=(120,180), 
		num_frames=100, pool_size=5, subprocess_threads=3):
	""" Applies multiprocessing to parallelize the retrieval of `num_videos` 
	videos using the `get_random_video()` function above. 
	""" 

	call_list = [video_list, prefix,0, output_size, num_frames, subprocess_threads]

	with Pool(pool_size) as p:
		videos = p.starmap(print_args, [call_list for i in range(num_videos)])
	# pdb.set_trace()	
	return videos

def get_random_small_batch(num_videos, video_list, prefix="", output_size=(120,180), 
		num_frames=100, subprocess_threads=3):
	""" This gets a SMALL batch (i.e., few batch elements) without using top-
	level multiprocessing. The multithreading is dedicated to reading in the 
	long video files. 
	"""	
	call_list = [video_list, prefix,0, output_size, num_frames, subprocess_threads]

	videos = [print_args(*call_list) for i in range(num_videos)]
	# pdb.set_trace()	
	return videos


	





NUM_FRAMES = 3000
RESOLUTION = (360, 540)
BATCH_SIZE = 1
POOL_SIZE = 0
SUB_POOL_SIZE = 10

NUM_REPEATS = 5
print("\n\n\n")

for i in range(NUM_REPEATS):
	start = time.time()
	batch_example = get_random_small_batch(BATCH_SIZE, mp4_list, prefix=DATA_FOLDER, output_size = RESOLUTION, 
			num_frames = NUM_FRAMES, subprocess_threads=SUB_POOL_SIZE)

	batch_np_tensor = np.concatenate(batch_example, axis=0)

	end = time.time()
	print(f"Time to retrieve videos: \t\t\t\t{end-start}")
	print("\tVideo shape: ", batch_np_tensor.shape)




