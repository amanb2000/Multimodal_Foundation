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
from packaging import version

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'