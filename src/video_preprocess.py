"""
Functions for pre-processing the video Dataset into usable spacetime patches
with Fourier features for positional encoding. 

Preliminaries: `video_loader` functions to load a video Dataset. 

Prototyped in `/sketch/03_Transformer_Preliminaries.ipynb`. 
"""

## Import box 
import os 
import sys 
import random
import pathlib
import itertools
import collections
import math

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pathlib 
from tqdm import tqdm
import cv2
# Some modules to display an animation using imageio.
import imageio
from urllib import request
# from tensorflow_docs.vis import embed

import video_loader as vl 

import pdb


def make_patchset(VideoSet, patch_duration, patch_height, patch_width): 
	"""Takes a VideoSet tf.data.Dataset object with full video tensors 
	of size [batch, nframes, height, width, channels] and returns a set of 
	patches. 

	These are NOT flattened patches. 
	"""
	def create_patches(video_tensor, ksizes=[1,4,16,16,1]): 
		"""Converts `video_tensor` into a sequence of patches.
		Input has shape [frames, width, height, channels]

		ksizes=[1,4,16,16,1] -> 4 temporal depth, 16 height/width. 
		"""
		return tf.extract_volume_patches(video_tensor, ksizes,ksizes,"VALID")

	ks = [1, patch_duration, patch_height, patch_width, 1]

	PatchedSet = VideoSet.map(lambda x: create_patches(x, ksizes=ks))
	return PatchedSet


def patch_to_flatpatch(PatchedSet, batch_size=1): 
	""" Takes a PatchedSet and flattens the [x, y, time] dimensions so that 
	each video is a 2D matrix of patches. Does not account for positional encoding. 
	"""
	def flatten_patched(patch_tensor, batch_size=1):
		""" Flattens the 3D structure of spacetime patches into a [batch, num_patches, patch_len] 
		tensor. Should be applied after positional encoding, generally. 
		"""
		print("Flattening a tensor of shape: ", patch_tensor.shape)
		b, t, h, w, cs = patch_tensor.shape
		return tf.reshape(patch_tensor, [batch_size, t*h*w, cs])

	FlatPatchSet = PatchedSet.map(lambda x: flatten_patched(x, batch_size=batch_size))
	return FlatPatchSet


def add_spacetime_codes(patch_tensor, k_space=15, mu_space=20, 
		k_time=64, mu_time=200):
	"""Given an unflattened patch_tensor, this returns the same patch tensor 
	with each patch CONCATENATED with x/y/time fourier codes according to 
	k, mu values specified in kwargs. 

	`k_space/time`: 	Number of frequency bands for the positional encoding. 
	`mu_space/time`: 	Nyquist frequency for each positional encoding. I.e., 
						mu/2 is the maximum frequency -> we can resolve 
						positional differences at up to mu/2 frequency.
	"""
	_, n_times, n_heights, n_widths, _ = patch_tensor.shape
	codes = fourier_tensor = get_spacetime_codes(n_times, n_heights, n_widths,
		k_space=k_space, mu_space=mu_space, k_time=k_time, mu_time=mu_time)
	
	codes = tf.expand_dims(codes, 0)

	return tf.concat([patch_tensor, codes], axis=-1)




## Helper Functions
def unflatten_patched(patch_tensor, duration, height, width):
	b, thw, cs = patch_tensor.shape
	assert thw == duration*height*width, "Patch duration*height*width must equal the patch's flattened length!"
	return tf.reshape(patch_tensor, duration, height, width, cs) 

def get_spacetime_codes(num_times, num_heights, num_widths, k_space=15, 
		mu_space=20, k_time=64, mu_time=200):
	""" Produces Fourier features for positionally coding x, y, and time 
	coordinates. Separate k,mu values for space and time. 
	
	Returns a tensor to be concatenated with the unflattened patch dataset 
	of shape [1,num_times, num_heights, num_widths, k_space*2*2 + k_time*2 + 3]
	"""
	## 1: Making the x_m array for each. 
	x_m_t = (tf.range(num_times, dtype=tf.float32) / num_times) * 2 - 1
	x_m_h = (tf.range(num_heights, dtype=tf.float32) / num_heights) * 2 - 1
	x_m_w = (tf.range(num_widths, dtype=tf.float32) / num_widths) * 2 - 1

	## 2: Making the f_k's for each. 
	space_fks = tf.range(1, mu_space/2, delta=(mu_space/2 - 1)/k_space, dtype=tf.float32)
	time_fks = tf.range(1, mu_time/2, delta=(mu_time/2 - 1)/k_time, dtype=tf.float32)

	## 3: time, height, and width sine arguments. 
	#  These have dimensions [num positions, numbands]
	t_args = math.pi * tf.expand_dims(x_m_t, axis=1) * tf.expand_dims(time_fks, axis=0)
	h_args = math.pi * tf.expand_dims(x_m_h, axis=1) * tf.expand_dims(space_fks, axis=0)
	w_args = math.pi * tf.expand_dims(x_m_w, axis=1) * tf.expand_dims(space_fks, axis=0)

	# [num positions, num bands]
	t_sin = tf.sin(t_args) 
	h_sin = tf.sin(h_args)
	w_sin = tf.sin(w_args) 
	t_cos = tf.cos(t_args) 
	h_cos = tf.cos(h_args)
	w_cos = tf.cos(w_args) 

	# Combined sin/cos/x_m features 
	t_feats = tf.concat( [t_sin, t_cos, tf.expand_dims(x_m_t, axis=1)], axis=1)
	h_feats = tf.concat( [h_sin, h_cos, tf.expand_dims(x_m_h, axis=1)], axis=1)
	w_feats = tf.concat( [w_sin, w_cos, tf.expand_dims(x_m_w, axis=1)], axis=1)

	# Expanding the dimensions and repeating -- [t,h,w,guh] order
	t_feats = tf.expand_dims(t_feats, 1)
	t_feats = tf.repeat(t_feats, num_widths, axis=1) # eventually width
	t_feats = tf.expand_dims(t_feats, 1)
	t_feats = tf.repeat(t_feats, num_heights, axis=1)

	h_feats = tf.expand_dims(h_feats, 0) 
	h_feats = tf.repeat(h_feats, num_times, axis=0) # directly to time
	h_feats = tf.expand_dims(h_feats, 2)
	h_feats = tf.repeat(h_feats, num_widths, axis=2) # directly to width

	w_feats = tf.expand_dims(w_feats, 0)
	w_feats = tf.repeat(w_feats, num_heights, axis=0) # eventually height 
	w_feats = tf.expand_dims(w_feats, 0)
	w_feats = tf.repeat(w_feats, num_times, axis=0) # directly time

	# Repeating each feature set along the proper dimensions 
	


	## 4: Gluing everything together -> fourier feature tensor 
	#  Each position in the tensor has size: (k_space*4) + k_time * 2 + 3
	# 	 - 2 * k_space for sin/cos of height 
	# 	 - 2 * k_space for sin/cos of width 
	# 	 - 2 * k_time for sin/cos of time 
	# 	 - 1 for x_m_t, 1 for x_m_h, 1 for x_m_w -- the indices \in [-1,1] 
	feature_dim = k_space * 4 + k_time * 2 + 3 

	fourier_tensor = tf.concat([t_feats, h_feats, w_feats], axis=-1)

	assert feature_dim == fourier_tensor.shape[-1]

	return fourier_tensor


# TODO: Finish the function to completely restore the original video tensor from 
# 		a set of patches. 


if __name__ == "__main__": 
	## Getting the VideoSet
	print("Getting VideoSet...")
	num_videos=10
	num_frames=20
	output_size=(240,360)
	VideoSet = vl.get_videoset("../datasets/downloads", num_videos, num_frames, output_size=output_size)
	print("Retrieved VideoSet!\n")

	## Getting patch dataset
	p_dur = 3
	p_height = 16
	p_width = 16
	print("Making patches from Videoset...")
	PatchSet = make_patchset(VideoSet, p_dur, p_height, p_width)
	print("Done making patches!")

	## Getting flat patch set 
	batch_size=1
	print("Making the flat patch set...")
	FlatPatchSet = patch_to_flatpatch(PatchSet, batch_size=batch_size)
	print("Done making the flat patch set!")

	## Adding codes 
	k_space = 15
	mu_space = 20 
	k_time = 64 
	mu_time = 200


	print("Adding codes to the PatchSet...")
	CodedPatchedSet = PatchSet.map(lambda x: add_spacetime_codes(x, 
			k_space=k_space, mu_space=mu_space, k_time=k_time, mu_time=mu_time))
	print("Done adding codes!\n")
	print("Flattening the coded + patched dataset...")
	FlatCodedPatchedSet = patch_to_flatpatch(CodedPatchedSet, batch_size=batch_size)
	print("Done flattening!\n")
	print("All done! Report: ")
	print("VideoSet: ", VideoSet)
	print("PatchSet: ", PatchSet) 
	print("FlatPatchSet: ", FlatPatchSet) 
	print("CodedPatchedSet: ", CodedPatchedSet)
	print("FlatCodedPatchedSet: ", FlatCodedPatchedSet)

