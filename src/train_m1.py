""" # Train model M1

This file contains a function for a custom training loop to train the 
perceiver-AE model M1. 

## Parameters

Big:  
 1. Model object 
 2. Dataset object

Hyperparams:
 1. Epochs
	 - `dset_size`: Can't tell how many training examples to use per epoch with
	 looping generator!
 2. Iterations per batch: Number of exposures the model gets over each video. 
	Each exposure samples from an input window (3) that moves along the frame. 
	The first exposure is from frame [0:windowsize], the last from [-windowsize:].
 3. Input window size (in frames). 
	 - The step increment is determined based on the `iterations per epoch`. 
 4. Test window size (in frames). 
	 - Step increment same as above. 
 5. Frequency of model checkpointing. 
"""
import tensorflow as tf
from tqdm import tqdm
import pdb


def training_step(model, video, optimizer, n_iters=3):
	""" We are just going to try and autoencode the video tensor. 
	That's literally it. 
	"""

	loss = 0.0 
	with tf.GradientTape() as tape:
		for i in range(n_iters):
			reset = i==0
			loss += model(video, reset_latent=reset)

	grads = tape.gradient(loss, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	return loss

	
	


def train_model(model, dataset, optimizer, dset_size, num_epochs=1, iterations_per_batch=33, 
		in_window_size=100, test_window_size=500):
	# TODO: Set up "testing loss" stuff

	all_losses = []
	total_loss_per_element = []

	for epoch in range(num_epochs):
		print(f"Epoch no. {epoch}")

		ecnt=0 # element count for this epoch
		for element in dataset:
			ecnt+=1
			print(f" ** Dataset element {ecnt}")
			if ecnt == dset_size:
				break

			element_loss = 0.0

			num_tokens = element.shape[1]
			print("\tnum tokens: ", num_tokens)
			window_start_inc = num_tokens/iterations_per_batch
			print("\tIncrement per sub iter: ", num_tokens)

			with tf.GradientTape() as tape:
				tape.watch(model.trainable_weights)
				for i in tqdm(range(iterations_per_batch)):
					window_start = int(window_start_inc*i)
					window_end = int(window_start + window_start_inc)
					if window_end > num_tokens: 
						break

					print(f"WINDOW: \t{window_start} -> {window_end}")

					model_input = element[:,0:50,:]
					print(f"Model input shape: {model_input.shape}")
					reset_model = i==0
					surprise = model(model_input, reset_latent=reset_model)
					print("Surprise: ", surprise.numpy())

					all_losses.append(surprise)
					element_loss += surprise

				total_loss_per_element.append(element_loss)
			
			grads = tape.gradient(element_loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			return
				
