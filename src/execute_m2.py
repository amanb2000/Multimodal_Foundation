""" This is a training script for the ViT-MAE inspired autoencoding vs. 
predictive coding experiment. 

Two perceiver autoencoders are trained to form representations of videos. 
Each receives the same frame(s) corresponding to the `present`. Some portion 
are masked, and these are used to form the latent representations. Both attempt 
to reconstruct the full set of `present` frame(s). 

One model is also tasked with predicting the pixel values of some `future` 
frames. These are sampled from the next `future` frames directly after the 
`present` time window. 

Once the models are trained, a separate script will be used to determine their 
effectiveness in forming representations of the Kinetics dataset (with linear
probing as a preliminary metric).
"""

## Import Box.
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
from packaging import version

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


## Parsing commandline arguments before importing TensorFlow.
print("\n\n\n")
print("\t=======================================")
print("\t=== 1 PARSING COMMANDLINE ARGUMENTS ===")
print("\t=======================================")

parser = argparse.ArgumentParser(description='Train model M1 (perceiver AE).')

parser.add_argument('--output-folder', action='store', type=str,
		help='Folder to keep checkpoints, loss plots, records of arguments, ' +
		'etc. The folder will be created if it doesn\'t exist. To make a ' + 
		'folder with the current date of format `2022-10-26_12:30:33`, add ' + 
		'`{now}` to the end of the string.\nDefault=../training/debug/{now}.',
		default='../training/debug/{now}')



## Dataset parameters 
parser.add_argument('--data-folder', action='store', type=str,
		help='Folder containing the `.mp4` video files to use for the ' +
		'experiment. Default = `../datasets/downloads`.', 
		default='../datasets/downloads')

parser.add_argument('--frame-size', action='store', type=str, 
		help='`height,width` of each video frame. Default=`120,180`.',
		default='120,180')

parser.add_argument('--patch-hwd', action='store', type=str, 
		help='`patch_height,patch_width,patch_duration`. Default=`16,16,3`.', 
		default='16,16,3')

parser.add_argument('--batch-size', action='store', type=int,
		help='Size of the batch (number of video tensors per training batch). '+
		'Default=10.',
		default=10)

parser.add_argument('--num-prefetch', action='store', type=int, 
		help='Number of dataset batches that are pre-fetched. Default=4.',
		default=4)

parser.add_argument('--k-mu-space', action='store', type=str, 
		help='`k,mu` for spatial Fourier codes. Default=`15,20`.',
		default="15,20")

parser.add_argument('--k-mu-time', action='store', type=str,
		help='`k,mu` for temporal Fourier codes. Default=`64,200`.',
		default='64,200')

## Video loader args
parser.add_argument('--pool-size', action='store', type=int, 
		help='Number of CPU threads allocated to the video loading.',
		default=20)

parser.add_argument('--threads-per-vid', action='store', type=int, 
		help='Number of threads for loading an individual video. Only '+
		'use this for long videos.', 
		default=1)


## Training parameters
parser.add_argument('--overfit', action='store', type=int, default=-1, 
		help='Take the first `n` videos from `mp4list` and overfit the model ' + 
		'on those. Default=-1 (i.e. use the full dataset).')

parser.add_argument('--cpu-only', action='store_true', 
		help='Include this flag to force training to use only CPU.')

parser.add_argument('--one-gpu', action='store_true', 
		help='Include this flag to force training to use only one GPU.')

parser.add_argument('--ckpt-period', action='store', type=int,
		help='Number of iterations separating each checkpoint. Default=50.',
		default=50)

parser.add_argument('--num-iters', action='store', type=int,
		help="Number of total iterations. Each iteration is one training step "+
		"on a batch of `--batch-size` videos each with `--num-frames`. Default=200.",
		default=200)

parser.add_argument('--lr', action='store', type=float,
		help="Primary optimizer learning rate. Default=0.001",
		default=0.001)


## Data traversal training parameters
parser.add_argument("--mask-ratio", action='store', type=float, default=0.3, 
		help='Portion of present tokens actually retained as model input.')

parser.add_argument("--alpha", action='store', type=float, default=0.7, 
		help='Weighting between present timewindow autoencoding vs. far '+ 
		'future prediction errors. 1 -> only present, 0 -> only future. '+
		'Default=0.7')

parser.add_argument("--present", action='store', type=int, default=1, 
		help="Number of frames in the `present` time window. Default=1.")

parser.add_argument("--future", action='store', type=int, default=3,
		help="Number of frames in the `future` time window. Default=3.")

parser.add_argument("--future-selection-probability", action='store', type=float,
		default=0.1, help="Probability of a token from the future time window "+
		"being selected. Default=0.33.")


## Model parameters
parser.add_argument('--restore-from', action='store', type=str, default=None,
		help='Path to an experiment directory. We will look at the '+
		'`checkpoints` subdirectory and start from the most recent one.'+
		' default=None (i.e., instantiate a new model).')

parser.add_argument('--latent-dims', action='store', type=str, default='100,700', 
		help='Dimensions of the latent state/predictive code tensor in the '+
		'model. Comma separated `num_tokens,token_dim`. Default=`100,700`')

parser.add_argument('--nheads', action='store', type=int, default=15, 
		help="Number of heads in each transformer block. Default=15")

parser.add_argument('--keydim', action='store', type=int, default=15, 
		help="Dimension of each key in each transformer ehad. Default=15.")

parser.add_argument('--mhadropout', action='store', type=float, default=0.0, 
		help="Dropout rate for multihead attention. Default=0.")

# Encoder
parser.add_argument('--n-enc-blocks', action='store', type=int, default=1, 
		help="Number of encoder blocks in the model. Default=1.")

# Latent evolver
parser.add_argument('--n-latent-blocks', action='store', type=int, default=5, 
		help="Number of transformer blocks in the latent module. Default=5.")
parser.add_argument('--identical-latent', action='store_true', 
		help="Include this flag to make each latent block identical.")

# Decoder 
parser.add_argument('--n-dec-blocks', action='store', type=int, default=3,
		help='Number of transformer blocks in the decoder module.')
parser.add_argument('--dec-expansion-block', action='store', type=int, default=2,
		help='Block number (0-indexed) when the token dimensionality is '+
		'expanded in the decoder.')



args = parser.parse_args() 
print("ARGS: \n\t", args)

# TODO: num_frames = present + future (done)
# 		distinct_latent -> identical_latent
# 		p_droptoken 	-> mask_ratio
#
#		no_redroptoken 	-> DNE



print("\n\n\n")
print("\t================================================")
print("\t=== SETTING UP OUTPUT FOLDER/LOGGING/ARG VAL ===")
print("\t================================================")


## Output folder for experiment information.
NOW = None
if args.output_folder.endswith('{now}'):
	args.output_folder = args.output_folder[:-5]
	print(args.output_folder)
	ct = str(datetime.datetime.now()).replace(' ', '_')
	ct = ct.split('.')[0]
	print("\n",ct)
	NOW = ct
	args.output_folder = os.path.join(args.output_folder, ct)
if not os.path.exists(args.output_folder):
	os.makedirs(args.output_folder)


## Setting up logging (screen + log files).
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


## Validating args
# Checking data 
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

# Checkpoint restoration 
assert args.restore_from == None or os.path.exists(args.restore_from), f"Checkpoint folder DNE: `{args.restore_from}`."

## Import Box II
# Updating CPU environment variable before importing Tensorflow.
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
import parallel_video_loader as pvl

## Reducing GPU's to 1 if needed. 
if args.one_gpu:
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		# Restrict TensorFlow to only use the first GPU
		try:
			tf.config.set_visible_devices(gpus[1], 'GPU')
			logical_gpus = tf.config.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
		except RuntimeError as e:
			# Visible devices must be set before GPUs have been initialized
			print(e)



## Getting the GPU set up
print("\n\n\n")
print("\t========================")
print("\t=== GPU DEVICE SETUP ===")
print("\t========================")

physical_devices = tf.config.list_physical_devices("GPU")
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
print("Physical devices: ",physical_devices)





print("\n\n\n")
print("\t=============================")
print("\t=== DATA FORMAT ARG PARSE ===")
print("\t=============================")
## Checkpoint args/restore from 
latest = None
if args.restore_from != None:
	latest = tf.train.latest_checkpoint(os.path.join(args.restore_from, 'checkpoints'))


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
	print("Restoring data args from checkpoint folder...")
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
num_frames = args.present+args.future

if args.overfit != -1:
	mp4_list = mp4_list[:args.overfit]



print("DATASET META INFO")
print("\tnum_frames: ", num_frames)
print("\tbatch_size: ", batch_size)
print("\toutput_size: ", output_size)
print("\tPatch h/w/d: ", patch_height, patch_width, patch_duration)
print("\tk, mu for space, time: ", (k_space, mu_space), (k_time, mu_time))
print("\tLatent dims: ", latent_dims)

if args.restore_from != None: 
	print("\tRestoring from -- ", args.restore_from)
	print("\tMost recent checkpoint: ", latest)

if args.overfit != -1:
	print(f"\tOverfitting to the first {args.overfit} elements.")
else: 
	print("\tmp4_list[:10] -- ", mp4_list[:10])


print("\n\n\n\t==========================")
print("\t=== VIDEO LOADER SETUP ===")
print("\t==========================")


DATA_FOLDER = "../datasets/downloads"
path_list = [os.path.join(DATA_FOLDER, i) for i in mp4_list]

vid_generator = pvl.get_generator(path_list, output_size, num_frames, batch_size, 
			pool_size=args.pool_size, thread_per_vid=args.threads_per_vid)

