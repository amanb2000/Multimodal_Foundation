""" Training script for model 1. 

GOAL: Leverage `generator` for custom `Dataset` object, perform profiling to 
determing the effectiveness of prefetching.
"""

## Import Box 
import os 
import sys 
import random
import pathlib
import itertools
import collections
import math
import argparse
import datetime
import pickle
import pdb


## Parsing commandline arguments -- must occur here because tensorflow 
#  imports may change based on `--cpu-only` flag.
print("\n\n\n")
print("\t=====================================")
print("\t=== PARSING COMMANDLINE ARGUMENTS ===")
print("\t=====================================")


parser = argparse.ArgumentParser(description='Train model M1 (perceiver AE).')

parser.add_argument('--output-folder', action='store', type=str,
		help='Folder to keep checkpoints, loss plots, records of arguments, ' +
		'etc. The folder will be created if it doesn\'t exist. To make a ' + 
		'folder with the current date of format `2022-10-26_12:30:33`, add ' + 
		'`{now}` to the end of the string.\nDefault=../training/debug/{now}.',
		default='../training/debug/{now}')



## Dataset parameters ##
parser.add_argument('--data-folder', action='store', type=str,
		help='Folder containing the `.mp4` video files to use for the ' +
		'experiment.', 
		default='../datasets/downloads')

parser.add_argument('--num-frames', action='store', type=int,
		help='Number of frames to gather from each video for the dataset. '+
		'Default=100.',
		default=100)

parser.add_argument('--frame-size', action='store', type=str, 
		help='`height,width` of each video frame. Default=120,180',
		default='120,180')

parser.add_argument('--patch-hwd', action='store', type=str, 
		help='`patch_height,patch_width,patch_duration`. Default=16,16,3', 
		default='16,16,3')

parser.add_argument('--batch-size', action='store', type=int,
		help='Size of the batch (number of video tensors per training batch). '+
		'Default=10',
		default=10)


parser.add_argument('--num-prefetch', action='store', type=int, 
		help='Number of dataset batches that are pre-fetched. Default=4.',
		default=4)

parser.add_argument('--k-mu-space', action='store', type=str, 
		help='`k,mu` for spatial Fourier codes. Default=15,20.',
		default="15,20")

parser.add_argument('--k-mu-time', action='store', type=str,
		help='`k,mu` for temporal Fourier codes. Default=64,200',
		default='64,200')



## Training parameters
parser.add_argument('--overfit', action='store', type=int, default=-1, 
		help='Take the first `n` videos from `mp4list` and overfit the model ' + 
		'on those. Default=-1 (i.e. use the full dataset)')
parser.add_argument('--cpu-only', action='store_true', 
		help='Include this flag to force training to use only CPU.')

parser.add_argument('--ckpt-period', action='store', type=int,
		help='Number of iterations separating each checkpoint. Default=50',
		default=50)

parser.add_argument('--num-iters', action='store', type=int,
		help="Number of total iterations. Each iteration is one training step "+
		"on a batch of `--batch-size` videos each with `--num-frames`. Default=200",
		default=200)

parser.add_argument('--lr', action='store', type=float,
		help="Primary optimizer learning rate. Default=0.001",
		default=0.001)





## Data traversal training parameters
parser.add_argument("--alpha", action='store', type=float, default=0.7, 
		help='Weighting between present timewindow prediction vs. far '+ 
		'future prediction errors. 1 -> only present, 0 -> only future. '+
		'Default=0.7')

parser.add_argument("--blind-iters", action='store', type=int, default=1, 
		help="Number of exposures at the beginning of procesing a video " + 
		"sequence that is NOT counted towards loss.")

parser.add_argument("--present", action='store', type=int, default=5, 
		help="Number of frames in the `present` time window. Default=5.")

parser.add_argument("--future", action='store', type=int, default=30,
		help="Number of frames in the `future` time window. Default=30.")

parser.add_argument("--future-selection-probability", action='store', type=float,
		default=0.1, help="Probability of a token from the future time window "+
		"being selected. Default=0.1")

parser.add_argument("--window-inc", action='store', type=int, default=5,
		help="How much the `present` and `future` time windows are incremented "+
		"each iteration in FRAMES. Default=5.")




## Model parameters
parser.add_argument('--restore-from', action='store', type=str, default=None,
		help='Path to an experiment directory. We will look at the '+
		'`checkpoints` subdirectory and start from the most recent one.')

parser.add_argument('--latent-dims', action='store', type=str, default='700,100', 
		help='Dimensions of the latent state/predictive code tensor in the '+
		'model. Comma separated `num_tokens,token_dim`. Default=700,10')

parser.add_argument('--nheads', action='store', type=int, default=15)

parser.add_argument('--keydim', action='store', type=int, default=15)

parser.add_argument('--mhadropout', action='store', type=float, default=0.0)



# Encoder
parser.add_argument('--n-enc-blocks', action='store', type=int, default=3, 
		help="Number of encoder blocks in the model. Default=3.")

parser.add_argument('--p-droptoken', action='store', type=float, default=0.5,
		help='Expected portion of input tokens retained on each exposure of ' + 
		'the latent state. Default=0.5.')

parser.add_argument('--no-re-droptoken', action='store_true', 
		help='Include this flag if you do NOT want the dropped tokens to be ' + 
		'resampled on each exposure.')



# Latent evolver
parser.add_argument('--n-latent-blocks', action='store', type=int, default=3, 
		help="Number of transformer blocks in the latent module.")
parser.add_argument('--distinct-latent', action='store_true', 
		help="Include this flag to make each latent block distinct (non-identical)")


# Decoder 
parser.add_argument('--n-dec-blocks', action='store', type=int, default=3,
		help='Number of transformer blocks in the decoder module.')
parser.add_argument('--dec-expansion-block', action='store', type=int, default=2,
		help='Block number (0-indexed) when the token dimensionality is '+
		'expanded in the decoder.')






args = parser.parse_args() 
print("ARGS: \n\t", args)

## Validating args
# Output folder
if args.output_folder.endswith('{now}'):
	args.output_folder = args.output_folder[:-5]
	print(args.output_folder)
	ct = str(datetime.datetime.now()).replace(' ', '_')
	ct = ct.split('.')[0]
	print("\n",ct)
	args.output_folder = os.path.join(args.output_folder, ct)
if not os.path.exists(args.output_folder):
	os.makedirs(args.output_folder)

# Setting up logging
class tee :
    def __init__(self, _fd1, _fd2) :
        self.fd1 = _fd1
        self.fd2 = _fd2

    def __del__(self) :
        if self.fd1 != sys.stdout and self.fd1 != sys.stderr :
            self.fd1.close()
        if self.fd2 != sys.stdout and self.fd2 != sys.stderr :
            self.fd2.close()

    def write(self, text) :
        self.fd1.write(text)
        self.fd2.write(text)

    def flush(self) :
        self.fd1.flush()
        self.fd2.flush()

stdoutsav = sys.stdout
out_log = open(os.path.join(args.output_folder, "stdout.log"), "w")
sys.stdout = tee(stdoutsav, out_log)

stderrsav = sys.stderr
err_log = open(os.path.join(args.output_folder, "stderr.log"), "w")
sys.stderr = tee(stderrsav, err_log)






# Data folder
assert os.path.exists(args.data_folder), f"Invalid data folder `{args.data_folder}` -- no directory!"
assert os.path.isdir(args.data_folder), f"Data folder `{args.data_folder}` is not a directory!"

# Training loop parameters
assert args.alpha >= 0 and args.alpha <= 1, f"Alpha must be between 0 and 1 -- received {args.alpha}"
assert args.blind_iters < args.num_frames/args.window_inc, f"Blind iterations exceeds total number of iterations!"
assert args.present < args.num_frames, f"Present window size mus not exceed the number of frames loaded per video!"
assert args.future < args.num_frames,  f"Future window size mus not exceed the number of frames loaded per video!"



# Frame size
out_size = args.frame_size.split(',')
assert len(out_size) == 2, f"Invalid `--frame-size` parameter: {args.frame_size}"
output_size = [int(i) for i in out_size]
# Patch size
patch_hwd = args.patch_hwd.split(',')
assert len(patch_hwd) == 3, f"Invalid `--patch-hwd` parameter: {args.patch_hwd}"
patch_height, patch_width, patch_duration = [int(i) for i in patch_hwd]



if args.cpu_only:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import imageio

import m1
import video_loader as vl
import video_preprocess as vp
import train_m1


# Restore from 
assert args.restore_from == None or os.path.exists(args.restore_from), f"Checkpoint folder DNE: `{args.restore_from}`."
latest = None

if args.restore_from != None:
	latest = tf.train.latest_checkpoint(os.path.join(args.restore_from, 'checkpoints'))

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

# Meta/constants
latent_dims = [int(i) for i in args.latent_dims.split(',')] # potentially overwritten in perceiver_kwargs
assert len(latent_dims) == 2, f"Invalid latent dims: `{args.latent_dims}`"

# Fourier parameters
_k_space, _mu_space = [int(i) for i in args.k_mu_space.split(',')]
_k_time, _mu_time = [int(i) for i in args.k_mu_time.split(',')]

## Data format args: dictionary holding all the arguments we use to format 
#  incoming video data that must match if we restore a model from a savepoint.
data_format_args = {
	'k_space': _k_space,
	'mu_space': _mu_space,
	'k_time': _k_time,
	'mu_time': _mu_time,
	'out_size': out_size,
	'patch_height': patch_height,
	'patch_width': patch_width, 
	'patch_duration': patch_duration
}

# Now we check if we need to load in the data format arguments.
if args.restore_from != None: 
	# Load data (deserialize)
	data_args_path = os.path.join(args.restore_from, 'data_format_args.pkl')
	with open(data_args_path, 'rb') as handle:
		data_format_args = pickle.load(handle)

# Now we unpack them, regardless of whether they were overwritten.
k_space = data_format_args['k_space']
mu_space = data_format_args['mu_space']
k_time = data_format_args['k_time']
mu_time = data_format_args['mu_time']
out_size = data_format_args['out_size']
patch_height = data_format_args['patch_height']
patch_width = data_format_args['patch_width']
patch_duration = data_format_args['patch_duration']

# Now we save these arguments for next time.
print("Saving data formatting arguments...")
data_args_path = os.path.join(args.output_folder, 'data_format_args.pkl')
with open(data_args_path, 'wb') as handle:
    pickle.dump(data_format_args, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")




# Non-model specific data parameters
mp4_list = os.listdir(args.data_folder)
batch_size = args.batch_size
num_prefetch = args.num_prefetch
num_frames = args.num_frames

if args.overfit != -1:
	mp4_list = mp4_list[:args.overfit]

print("\n\n\n")
print("\t=========================")
print("\t=== DATASET META INFO ===")
print("\t=========================")
print("num_frames: ", num_frames)
print("batch_size: ", batch_size)
print("output_size: ", output_size)
print("Patch h/w/d: ", patch_height, patch_width, patch_duration)
print("k, mu for space, time: ", (k_space, mu_space), (k_time, mu_time))
print("Latent dims: ", latent_dims)

if args.restore_from != None: 
	print("Restoring from -- ", args.restore_from)
	print("\tMost recent checkpoint: ", latest)

if args.overfit != -1:
	print(f"\n\n\nOVERFITTING TO FIRST {args.overfit} ELEMENTS!!!")
else: 
	print("mp4_list[:10] -- ", mp4_list[:10])




## Creating the generator
print("\n\n\n")
print("\t========================")
print("\t=== DATASET CREATION ===")
print("\t========================")
def get_generator(vid_list, dat_folder, out_size, n_frames):
	def generate_video_tensors():
		""" This is a generator for raw video tensors of shape 
		[num_frames, height, width, channels]. 

		It uses global variables defined above under "meta/constants". These 
		will be commandline arguments in the future. 
		"""
		_mp4_list = vid_list
		_DATA_FOLDER = dat_folder 
		_output_size = out_size 
		_num_frames = n_frames 
		while True: 
			for fname in _mp4_list:
				try:
					retval = vl.get_single_video_tensor(os.path.join(_DATA_FOLDER, fname), _num_frames, output_size=_output_size)
				except:
					continue
				if type(retval) == np.ndarray and retval.shape[0] == _num_frames:
					yield np.expand_dims(retval, axis=0)

	return generate_video_tensors

video_generator = get_generator(mp4_list, args.data_folder, output_size, num_frames)

videoset = tf.data.Dataset.from_generator(video_generator, output_signature=tf.TensorSpec(shape=[1, num_frames, *output_size, 3], dtype=tf.float32))
print(videoset)

def show_nn_sq(video_tensor, n=3, title="Some Frames", fname="bruh.png"):
	n2 = n*n
	fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)

	_nframes = video_tensor.shape[0]
	frame_inc = _nframes//(n2)

	for i in range(n):
		for j in range(n):
			full_idx = i*n + j

			axs[i,j].imshow(video_tensor[frame_inc*full_idx,:,:,:])
			axs[i,j].set_title(f"Frame {frame_inc*full_idx}")

	fig.suptitle(title)
	plt.savefig(fname)
	plt.close()

out_test = os.path.join(args.output_folder, "source_video_example.png")
print(f"Showing off an element of the videodataset(generator) in `{out_test}`")	
for element in videoset:
	show_nn_sq(tf.squeeze(element), fname=out_test)
	break


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

instantiation_params = None
if args.restore_from != None: 
	# Load data (deserialize)
	instant_param_pth = os.path.join(args.restore_from, 'instantiation_params.pkl')
	with open(instant_param_pth, 'rb') as handle:
		instantiation_params = pickle.load(handle)

## Setting up the component modules of the top-level PAE model.
# Encoder 

# Getting arguments 
encoder_args = None
encoder_kwargs = None

if instantiation_params != None: # if we are restoring, we need to use the exact same args.
	encoder_args = instantiation_params['encoder_args']
	encoder_kwargs = instantiation_params['encoder_kwargs']
else:
	n_encoder_blocks = args.n_enc_blocks
	encoder_tfres = False
	enc_nheads = args.nheads
	enc_keydim = args.keydim
	enc_mhadropout = args.mhadropout

	encoder_args = [n_encoder_blocks]
	encoder_kwargs = {
		'tfblock_residual': encoder_tfres, 
		'n_heads': enc_nheads, 
		'key_dim': enc_keydim, 
		'mha_dropout': enc_mhadropout
	}

test_encoder = m1.PAE_Encoder(*encoder_args, **encoder_kwargs)

# Latent evolver 
latent_ev_args = None
latent_ev_kwargs = None

if instantiation_params != None: 
	latent_ev_args = instantiation_params['latent_ev_args']
	latent_ev_kwargs = instantiation_params['latent_ev_kwargs']
else:	
	n_latentev_blocks = args.n_latent_blocks
	latent_distinct_blocks = args.distinct_latent
	latent_residual = False
	latent_nheads = args.nheads
	latent_keydim = args.keydim
	latent_mhadropout = args.mhadropout

	latent_ev_args = [n_latentev_blocks]
	latent_ev_kwargs = {
		'distinct_blocks': latent_distinct_blocks, 
		'tfblock_residual': latent_residual, 
		'n_heads': latent_nheads, 
		'key_dim': latent_keydim, 
		'mha_dropout': latent_mhadropout
	}

test_latent_ev = m1.PAE_Latent_Evolver(*latent_ev_args,**latent_ev_kwargs) 


# decoder
decoder_args = None
decoder_kwargs = None

if instantiation_params != None:
	decoder_args = instantiation_params['decoder_args']
	decoder_kwargs = instantiation_params['decoder_kwargs']
else:
	output_patch_dim = patch_duration * patch_height * patch_width * 3
	n_decoder_blocks = args.n_dec_blocks
	expansion_block_num = args.dec_expansion_block
	decoder_tfres = False
	dec_nheads = args.nheads
	dec_keydim = args.keydim
	dec_mhadropout = args.mhadropout

	decoder_args = [output_patch_dim, n_decoder_blocks, expansion_block_num]
	decoder_kwargs = {
		'tfblock_residual': decoder_tfres, 
		'n_heads': dec_nheads, 
		'key_dim': dec_keydim, 
		'mha_dropout': dec_mhadropout
	}

test_decoder = m1.PAE_Decoder(*decoder_args, **decoder_kwargs)

# Loss function definition
mse = tf.keras.losses.MeanSquaredError()

# Unpacking some params
p_droptoken = args.p_droptoken

## Instantiating the PAE model!
code_dim = 2*(2*k_space+1) + (2*k_time+1) # k_space = 15 and k_time = 64 -> 191

if instantiation_params != None: 
	perceiver_kwargs = instantiation_params['perceiver_kwargs']
else:
	perceiver_kwargs = {
		'code_dim': code_dim,
		'latent_dims': latent_dims,
		'p_droptoken': p_droptoken		
	}

perceiver_ae = m1.PerceiverAE(mse, test_encoder, test_latent_ev, test_decoder, **perceiver_kwargs)


## Saving arguments for instantiating the perceiver!
instantiation_params = {
	'encoder_args': encoder_args,
	'encoder_kwargs': encoder_kwargs,
	'latent_ev_args': latent_ev_args,
	'latent_ev_kwargs': latent_ev_kwargs,
	'decoder_args': decoder_args,
	'decoder_kwargs': decoder_kwargs,
	'perceiver_kwargs': perceiver_kwargs
}

instantiation_params_output_pth = os.path.join(args.output_folder, 'instantiation_params.pkl')
print(f"\nSaving the instantiation parameters to `{instantiation_params_output_pth}`")

# Store data (serialize)
with open(instantiation_params_output_pth, 'wb') as handle:
    pickle.dump(instantiation_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")


if args.restore_from != None: 
	print(f"Restoring model from {latest}!")
	perceiver_ae.load_weights(latest)

perceiver_ae.reset_latent()


print("Done setting up perceiver_ae model!")




print("\t=========================================================")
print(f"\t=== TRAINING MODEL: AUTOENCODING FULL {num_frames} FRAME VIDEOS ===")
print("\t=========================================================")


# Calculating tokens per frame 
num_width = output_size[1] // patch_width
num_height = output_size[0] // patch_height

tokens_per_frame = (num_width * num_height)/patch_duration
print("Tokens per frame: ", tokens_per_frame)

dset_size = args.num_iters
checkpoint_period = args.ckpt_period
checkpoint_path = os.path.join(args.output_folder,"checkpoints/cp-{epoch:04d}.ckpt")
if not os.path.exists(os.path.dirname(checkpoint_path)):
	os.makedirs(os.path.dirname(checkpoint_path))



optimizer = keras.optimizers.Adam(learning_rate=args.lr)
# train_m1.train_model(perceiver_ae, FlatCodedPatchedSet, optimizer, dset_size)



## Predictive training args
predictive_training_args = [tokens_per_frame]
predictive_training_kwargs = {
	'alpha': args.alpha,
	'blind_iters': args.blind_iters,
	'present_time_window': args.present,
	'prediction_time_window': args.future, 
	'prob_prediction_select': args.future_selection_probability, 
	'window_increment': args.window_inc
}


losses = []
present_losses = []
future_losses = []

current_gpu_use = []
peak_gpu_use = []

with tf.device('/GPU:1'):
	cnt=0
	for el in tqdm(FlatCodedPatchedSet, total=dset_size):
		tf.config.experimental.reset_memory_stats('/GPU:1')
		# loss = train_m1.training_step(perceiver_ae, el, optimizer)
		surprise_loss, future_loss, total_loss = train_m1.predictive_training_step(perceiver_ae, el, optimizer, *predictive_training_args, **predictive_training_kwargs)
		losses.append(total_loss.numpy())
		present_losses.append(surprise_loss.numpy())
		future_losses.append(future_loss.numpy())
		# print("Loss: ", total_loss.numpy())

		cnt+=1 
		if cnt == dset_size:
			break
		elif cnt == 2:
			print(perceiver_ae.summary())
			print("PERCEIVER N, C: ", perceiver_ae.N, perceiver_ae.C)
			print("Latent shape: ", perceiver_ae.latent.shape)
			os.mkdir(os.path.join(args.output_folder, "full_model"))
			# perceiver_ae.save(os.path.join(args.output_folder, "full_model/iter2"))

		dct = tf.config.experimental.get_memory_info('/GPU:1')
		current_gpu_use.append(dct["current"]*0.000001)
		peak_gpu_use.append(dct["peak"]*0.000001)

		if cnt % checkpoint_period == 0:
			perceiver_ae.save_weights(checkpoint_path.format(epoch=cnt))


	
print("\nDONE TRAINING!!!")

print("\nPlotting autoencoding losses over time.")

plt.plot(losses, label=f'{args.alpha}-weighted')
plt.plot(present_losses, label=f'present')
plt.plot(future_losses, label=f'future')
plt.title(f"Surprises over Time -- Autoencoding {num_frames} Frames")
plt.legend()
plt.savefig(os.path.join(args.output_folder, "loss.png"))
plt.close()


print("\nPlotting autoencoding losses over time.")

plt.plot(current_gpu_use, label="current")
plt.plot(peak_gpu_use, label="peak")
plt.title("GPU Use per Iteration")
plt.legend()
plt.savefig(os.path.join(args.output_folder,"GPU_use.png"))
plt.close()


out_log.close()
err_log.close()
