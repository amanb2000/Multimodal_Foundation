""" Training script for model 1. 

GOAL: Leverage `generator` for custom `Dataset` object, perform profiling to 
determing the effectiveness of prefetching.
"""

## Import Box 
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys 
import random
import pathlib
import itertools
import collections
import math
import pdb

import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
# Some modules to display an animation using imageio.
import imageio

import m1

## Getting the GPU set up
print("\n\n\n")
print("\t========================")
print("\t=== GPU DEVICE SETUP ===")
print("\t========================")

physical_devices = tf.config.list_physical_devices("GPU")
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

print("Physical devices: ",physical_devices)


## Acquiring the dataset!
import video_loader as vl
import video_preprocess as vp

# Meta/constants -- TODO: These should be commandline arguments. 
DATA_FOLDER = "../datasets/downloads"
num_frames = 10
output_size = (120, 180)

patch_height = 16
patch_width = 16
patch_duration = 3

batch_size = 10
num_prefetch = 4

# Fourier feature codes 
k_space = 15
mu_space = 20 
k_time = 64 
mu_time = 200

# Mp4 list: 
mp4_list = os.listdir(DATA_FOLDER)

print("\n\n\n")
print("\t=========================")
print("\t=== DATASET META INFO ===")
print("\t=========================")
print("num_frames: ", num_frames)
print("batch_size: ", batch_size)
print("output_size: ", output_size)
print("Patch h/w/d: ", patch_height, patch_width, patch_duration)
print("k, mu for space, time: ", (k_space, mu_space), (k_time, mu_time))
print("mp4_list[:10] -- ", mp4_list[:10])


## Creating the generator
print("\n\n\n")
print("\t========================")
print("\t=== DATASET CREATION ===")
print("\t========================")
def generate_video_tensors():
	""" This is a generator for raw video tensors of shape 
	[num_frames, height, width, channels]. 

	It uses global variables defined above under "meta/constants". These 
	will be commandline arguments in the future. 
	"""
	_mp4_list = mp4_list
	_DATA_FOLDER = DATA_FOLDER
	_output_size = output_size
	_num_frames = num_frames
	while True: 
		for fname in _mp4_list:
			try:
				retval = vl.get_single_video_tensor(os.path.join(_DATA_FOLDER, fname), _num_frames, output_size=_output_size)
			except:
				continue
			if type(retval) == np.ndarray and retval.shape[0] == _num_frames:
				yield np.expand_dims(retval, axis=0)

videoset = tf.data.Dataset.from_generator(generate_video_tensors, output_signature=tf.TensorSpec(shape=[1, num_frames, *output_size, 3], dtype=tf.float32))
print(videoset)

def show_nn_sq(video_tensor, n=3, outname="breh.png"):
	n2 = n*n
	fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)

	_nframes = video_tensor.shape[0]
	frame_inc = _nframes//(n2)

	for i in range(n):
		for j in range(n):
			full_idx = i*n + j

			axs[i,j].imshow(video_tensor[frame_inc*full_idx,:,:,:])
			axs[i,j].set_title(f"Frame {frame_inc*full_idx}")

	fig.suptitle(f"Video Survey over {_nframes} Frames")
	plt.savefig(outname)
	plt.show()

"""
out_test = "debug/test_from_videoset.png"
print(f"Showing off an element of the videodataset(generator) in `{out_test}`")	
for element in videoset:
	show_nn_sq(tf.squeeze(element), outname=out_test)
	break
"""

print("Making patches from Videoset...")
PatchSet = vp.make_patchset(videoset, patch_duration, patch_height, patch_width)

print("Making the flat patch set...")
FlatPatchSet = vp.patch_to_flatpatch(PatchSet, batch_size=1)

print("Adding codes to the PatchSet...")
CodedPatchedSet = PatchSet.map(lambda x: vp.add_spacetime_codes(x, 
		k_space=k_space, mu_space=mu_space, k_time=k_time, mu_time=mu_time))

print("Flattening the coded + patched dataset...")
FlatCodedPatchedSet = vp.patch_to_flatpatch(CodedPatchedSet, batch_size=1)
FlatCodedPatchedSet = FlatCodedPatchedSet.map(lambda x: tf.squeeze(x))
FlatCodedPatchedSet = FlatCodedPatchedSet.batch(batch_size)
FlatCodedPatchedSet = FlatCodedPatchedSet.prefetch(num_prefetch)

cnt = 0
for el in FlatCodedPatchedSet: 
	print(" ** Shape of FlatCodedPatchedSet element: ", el.shape)
	cnt += 1
	if cnt == 2:
		break

print("Done getting datasets setup!")





## Setting up the model
print("\n\n\n")
print("\t========================")
print("\t=== SETTING UP MODEL ===")
print("\t========================")

## Setting up the component modules of the top-level PAE model.
# Encoder 
n_encoder_blocks = 3
p_droptoken = 0.3
re_droptoken = True
encoder_tfres = False
enc_nheads = 15
enc_keydim = 15
enc_mhadropout = 0.0
test_encoder = m1.PAE_Encoder(n_encoder_blocks, p_droptoken=p_droptoken, re_droptoken=re_droptoken, tfblock_residual=encoder_tfres, n_heads=enc_nheads, key_dim=enc_keydim, mha_dropout=enc_mhadropout)

# Latent evolver 
n_latentev_blocks = 3
latent_distinct_blocks = False
latent_residual = False
latent_nheads = 15
latent_keydim = 15
latent_mhadropout=0.0
test_latent_ev = m1.PAE_Latent_Evolver(n_latentev_blocks, distinct_blocks=latent_distinct_blocks, tfblock_residual=latent_residual, n_heads=latent_nheads, key_dim = latent_keydim, mha_dropout=latent_mhadropout)


# decoder
output_patch_dim = 2304
n_decoder_blocks = 3
expansion_block_num = 2
decoder_tfres = False
dec_nheads = 15
dec_keydim = 15
dec_mhadropout = 0.0
test_decoder = m1.PAE_Decoder(output_patch_dim, n_decoder_blocks, expansion_block_num, tfblock_residual=decoder_tfres, n_heads=dec_nheads, key_dim=dec_keydim, mha_dropout=dec_mhadropout)

# loss function
mse = tf.keras.losses.MeanSquaredError()

## Instantiating the PAE model!
perceiver_ae = m1.PerceiverAE(mse, test_encoder, test_latent_ev, test_decoder, code_dim=191)

perceiver_ae.reset_latent()


print("Done setting up perceiver_ae model!")


import train_m1


print("\t===================================================")
print(f"\t=== TRAINING MODEL: AUTOENCODING FULL {num_frames} VIDEOS ===")
print("\t===================================================")


dset_size = 500
checkpoint_period = 50
checkpoint_path = "../training/2022_10_25/cp-{epoch:04d}.ckpt"


optimizer = keras.optimizers.Adam(learning_rate=0.001)
# train_m1.train_model(perceiver_ae, FlatCodedPatchedSet, optimizer, dset_size)

losses = []
current_gpu_use = []
peak_gpu_use = []
with tf.device('/GPU:1'):
	tf.config.experimental.reset_memory_stats('/GPU:1')
	cnt=0
	for el in tqdm(FlatCodedPatchedSet, total=dset_size):
		loss = train_m1.training_step(perceiver_ae, el, optimizer)
		losses.append(loss.numpy())
		print("Loss: ", loss.numpy())



		cnt+=1 
		if cnt == dset_size:
			break
		elif cnt == 2:
			print(perceiver_ae.summary())

		dct = tf.config.experimental.get_memory_info('/GPU:1')
		current_gpu_use.append(dct["current"]*0.000001)
		peak_gpu_use.append(dct["peak"]*0.000001)

		if cnt % checkpoint_period == 0:
			perceiver_ae.save_weights(checkpoint_path.format(epoch=cnt))


	
print("\nDONE TRAINING!!!")

plt.plot(losses)
plt.title(f"Surprises over Time -- Autoencoding {num_frames} Frames")
plt.savefig("loss.png")
plt.close()

plt.plot(current_gpu_use, label="current")
plt.plot(peak_gpu_use, label="peak")
plt.title("GPU Use per Iteration")
plt.legend()
plt.savefig("GPU_use.png")
plt.close()
