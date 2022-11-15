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

# TODO: num_frames = present + future
# 		distinct_latent -> identical_latent
# 		p_droptoken 	-> mask_rate