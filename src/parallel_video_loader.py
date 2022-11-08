""" Based on the prototype in `dataset_benchmark.py`, I build a fully inter- 
and intra-parallelized video loader. 

In order to make the most of the available CPU's, the system will be able to 
parallelize loading a single video (i.e., multiple read heads) and also 
parallelize the reading of multiple video files. 
"""
import os 
import matplotlib.pyplot as plt
import pdb
from multiprocessing import Pool
import random
import time

# import tensorflow as tf
import cv2
import numpy as np


def load_video_range(video_path, start_frame, num_frames, output_size):
	""" Loads video from `video_path` starting on `start_frame` for 
	`num_frames` frames at `output_size` resolution. 

	Returns a numpy tensor of shape [1, num_frames, height, width, 3].
	"""
	result = [] # List of frames, converted to a numpy tensor at the end.

	src = cv2.VideoCapture(str(video_path))

	src.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	for i in range(num_frames):
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
			print("Unable to read frame within range in `load_video_range()` -- returning None!")
			return None

	
	src.release()

	if len(result) == 0:
		print(f"Unable to read in any frames in {video_path}. Check that the video path is valid.")
		return None

	result = np.concatenate(result)
	result = np.expand_dims(result,0)

	return result
	

def get_single_path_and_start(path_list, num_frames):
	cnt = 0
	while cnt < 20: 
		cnt += 1
		potential_path = random.choice(path_list)
		if os.path.exists(potential_path): 
			src = cv2.VideoCapture(str(potential_path))
			video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
			src.release()
			if video_length > num_frames+2:
				max_start = video_length - num_frames-1
				return potential_path, random.randint(0, max_start)

	assert False

def get_nth_tensor(clip_list, tensor_num, tensor_segments): 
	start_idx = tensor_num*tensor_segments 
	end_idx = (tensor_num+1)*tensor_segments

	return np.concatenate(clip_list[start_idx:end_idx], axis=1)

def load_video_batch(path_list, num_frames, batch_size, output_size=(120,180), 
		pool_size=30, thread_per_video=3):
	
	## select `batch_size` videos from `file_list` and generate their `start_frames`
	print("\tGetting path starts...")
	start_ = time.time()
	with Pool(min(batch_size, pool_size)) as p:
		path_starts = p.starmap(get_single_path_and_start, [(path_list, num_frames) for i in range(batch_size)]) 

	## Generate `thread_per_video` calls for each of the `path_starts` entries
	call_args = []
	start_inc = num_frames/thread_per_video
	start_inc_int = int(start_inc) 
	for ps in path_starts: 
		pth, start = ps
		call_list = [ [pth, int(start + i*start_inc), start_inc_int, output_size] for i in range(thread_per_video)]
		call_args += call_list

	end_ = time.time()
	print("\tTOOK ", end_-start_)

	print("\tCalling starmap on `load_video_range`...")
	start = time.time()
	## Invoke all the calls
	with Pool(pool_size) as p: 
		video_clips = p.starmap(load_video_range, call_args)
	end = time.time()
	print("\tTOOK ", end-start)

	## Recombine all the call results
	# pdb.set_trace()

	if thread_per_video > 1: 
		start = time.time()
		# video_tensors = [np.concatenate(video_clips[i*thread_per_video:(i+1)*thread_per_video], axis=1) for i in range(batch_size)]
		with Pool(pool_size) as p: 
			video_tensors = p.starmap(get_nth_tensor, [ (video_clips, i, thread_per_video) for i in range(batch_size) ])
		end = time.time()
		print("\tTOOK ", end-start)
	else: 
		video_tensors = video_clips

	batch_tensor = np.concatenate(video_tensors, axis=0)
	return video_tensors 

def get_generator(path_list, out_size, n_frames):
	def generate_video_tensors():
		""" This is a generator for raw video tensors of shape 
		[num_frames, height, width, channels]. 

		It uses global variables defined above under "meta/constants". These 
		will be commandline arguments in the future. 
		"""
		_path_list = path_list 
		_DATA_FOLDER = dat_folder 
		_output_size = out_size 
		_num_frames = n_frames 
		while True: 
			# retval = vl.get_single_video_tensor(os.path.join(_DATA_FOLDER, fname), _num_frames, output_size=_output_size)
			batch_tensor = load_video_batch(_path_list, _num_frames, BATCH_SIZE, output_size=RESOLUTION, 
					pool_size=POOL_SIZE, thread_per_video=THREAD_PER_VID)
			batch_tensor = np.concatenate(batch_tensor, axis=0)

	return generate_video_tensors


if __name__ == "__main__":
	DATA_FOLDER = "../datasets/downloads"
	mp4_list = os.listdir(DATA_FOLDER)
	path_list = [os.path.join(DATA_FOLDER, i) for i in mp4_list]


	NUM_FRAMES = 100 
	BATCH_SIZE = 30
	RESOLUTION = (120,180)
	POOL_SIZE = 30
	THREAD_PER_VID = 1

	for k in range(100):
		start = time.time()
		batch_tensor = load_video_batch(path_list, NUM_FRAMES, BATCH_SIZE, output_size=RESOLUTION, 
				pool_size=POOL_SIZE, thread_per_video=THREAD_PER_VID)
		batch_tensor = np.concatenate(batch_tensor, axis=0)
		print("batch tensor: ", batch_tensor.shape)
		end = time.time()
		print(f"TOOK {end-start} SECONDS")


