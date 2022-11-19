""" # Training function for m2

This contains functions for performing a training step of the 2nd training
modality -- i.e., comparing predictive vs. autoencoding codes. 

"""


import tensorflow as tf 
import numpy as np 
from tqdm import tqdm 
import pdb


def training_step(model, present, present_sampled, future_sampled, optimizer, alpha=0.5, n_iters=2, blind_iters=1, use_future=True):
	""" PROTOTYPE SIMPLE AUTOENCODING TRAINING STEP. 
	
	We are just going to try and autoencode the video tensor. 
	That's literally it. 

	kwargs:
		`blind_iters`: 	Number of iterations we give the model "for free" 
						before its prediction error starts affecting loss. 
	"""

	loss = 0.0 
	with tf.GradientTape() as tape:
		blind_loss = model(present_sampled, reset_latent=True)
		cur_loss = model(present_sampled, remember_this=False)
		if use_future: 
			fut_loss = model(future_sampled, remember_this=False)
			loss += cur_loss*alpha + (1-alpha)*fut_loss
		else: 
			fut_loss = -1 
			loss += cur_loss

	grads = tape.gradient(loss, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	return loss, cur_loss, fut_loss, blind_loss