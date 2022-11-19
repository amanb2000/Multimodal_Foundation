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
import numpy as np
from tqdm import tqdm
import pdb


# @tf.function
def predictive_training_step(model, batch_tensor, optimizer, tokens_per_frame, 
		alpha=0.75, blind_iters=1, present_time_window=5, 
		prediction_time_window=50, prob_prediction_select=0.1, 
		window_increment=5):
	""" Full Training Loop Step: Predictive Coding

	This function takes one training step based on a batch of data. The 
	loss is minimized for a predictive coding task. Given a batch of 
	flattened/patched/coded video tensors, the model does the following: 

	 1. Using a small rolling time window, frames are sampled from the video. 
	 	The sampled frames are fed into the model, and the model returns its 
		prediction error on trying to predict the incoming data prior to 
		processing the new data. 

		The new data is then processed by the model (i.e., its information is 
		projected onto the model's latent state). 
	 2. Using a larger rolling time window, frames are sampled again. This time, 
	 	calculate the model's loss on predicting the far-flung future 
		information but we do not allow the model to incorporate the future 
		information into its latent state tensor. 
	
	Steps (1-2) are performed every iteration as the windows both move forward 
	through the video tensor. 

	One gradient step is taken when the model finishes traversing all frames 
	of the batch tensor. This optimization step is taken to minimize the 
	sum of the losses calculated in steps (1-2).

	NOTE: Sampling from the present time window is handled by the 

	Args: 
		model: 
			Differentiable tensorflow model that can perform predictive coding. 
			See `m1.py`.
		batch_tensor:
			Tensor of shape [batch, num_tokens, token_dim] where each element i 
			batch_tensor[i,:,:] is the flat/patch/coded representation of a 
			video tensor with its Fourier spacetime codes attached. The time 
			dimension should be the final flattened dimension (just use the 
			code from `video_preprocess.py`).
		optimizer: 
			Keras optimizer called on the model. 
		tokens_per_frame: 
			Number of tokens that comprise a single frame. 
			 = (num_width*num_height)/patch_dration 
			where 
				num_width  = width_px // patch_width
				num_height = height_px // patch_height
		
	Kwargs:
		alpha: 
			Float in [0,1] for weighting surprise vs. future loss. 

				alpha = 1 --> ONLY immediate surprise loss. 
				alpha = 0 --> ONLY future loss. 


		blind_iters: 
			Number of iterations *not* counted toward prediction or test loss 
			at the beginning of the tensor sweep. Corresponds to "taking a brief 
			glance around" before actually trying to perform predictive coding.
		present_time_window: 
			Time window for the "present". This information makes its way 
			onto the latent tensor. 	
		prediction_time_window: 
			[In frames] Time window for the "future". This information does not
			make its way onto the latent tensor, but loss on this 
		prob_prediction-select: 
			Probability that a given token within the prediction time window 
			will be selected for.
		window_increment: 
			[In frames] Number of frames to advance per iteration. We finish
			when the "present" window gets to the end of the frame. The
			future/prediction time window does not move once it hits the end. 


	Returns: 
		(surprise_loss, future_loss, total_loss):
			Totals over the course of training. 
	``
	"""

	assert 0 <= alpha

	num_tokens = batch_tensor.shape[1]

	num_frames = round(num_tokens/tokens_per_frame)
	window_inc_tokens = window_increment * tokens_per_frame
	num_iters = round(num_tokens/window_inc_tokens)

	window_inc_tokens = window_inc_tokens

	present_window_tokens = round(present_time_window * tokens_per_frame)
	future_window_tokens = round(prediction_time_window * tokens_per_frame)

	"""
	print("\nNUM TOKENS: ", num_tokens)
	print("NUM FRAMES: ", num_frames)

	print("Window increment (frames): ", window_increment)
	print("Window increment (tokens): ", window_inc_tokens)

	print("NUM ITERS: ", num_iters)
	print("Future window tokens (pre-mask): ", future_window_tokens)
	"""

	future_msk = np.zeros(future_window_tokens, dtype=np.bool)
	future_msk[:round(future_window_tokens*prob_prediction_select)] = True
	np.random.shuffle(future_msk)

	surprise_loss = 0.0 	# Corresponds to prediction of incoming data from the "present"
	future_loss = 0.0		# Corresponds to prediction loss on far future data. 
	loss = 0.0

	with tf.GradientTape() as tape:
		for i in range(num_iters):
			# Calculating the present window.
			present_window_end = i*window_inc_tokens + present_window_tokens 
			present_window_end = round(present_window_end)
			if present_window_end > num_tokens:
				present_window_end = num_tokens
			window_start = present_window_end - present_window_tokens

			# Calculating the future window.
			future_window_end = window_start + future_window_tokens
			if future_window_end > num_tokens:
				future_window_end = num_tokens

			future_window_start = future_window_end - future_window_tokens
			# print(f"Iteration {i} of {num_iters}")
			# print(f"\tPRESENT WINDOW: [{window_start}, {present_window_end}] -- \t{present_window_end-window_start}")
			# print(f"\tFUTURE WINDOW: [{future_window_start}, {future_window_end}] -- \t{future_window_end - future_window_start}")

			## Extracting current windowed tensors 
			present = batch_tensor[:,window_start:present_window_end,:]
			future_ = batch_tensor[:,future_window_start:future_window_end,:]
			future = tf.boolean_mask(future_, future_msk, axis=1)

			np.random.shuffle(future_msk)

			# print("\tPresent Tensor Shape: ", present.shape)
			# print("\tFuture  Tensor Shape: ", future.shape)

			reset = i==0

			cur_surprise = model(present, reset_latent=reset)
			fut_surprise = model(future, remember_this=False, no_droptoken=True)
			if i >= blind_iters:
				surprise_loss += cur_surprise
				future_loss += fut_surprise		


		surprise_loss /= num_iters
		future_loss /= num_iters

		loss = alpha*surprise_loss + (1-alpha)*future_loss


	print("Total surprise: ", surprise_loss.numpy())
	print("Total future loss: ", future_loss.numpy())

	print("Final loss: ", loss.numpy())

	grads = tape.gradient(loss, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))

	return surprise_loss, future_loss, loss
	




def training_step(model, video, optimizer, n_iters=3, blind_iters=1):
	""" PROTOTYPE SIMPLE AUTOENCODING TRAINING STEP. 
	
	We are just going to try and autoencode the video tensor. 
	That's literally it. 

	kwargs:
		`blind_iters`: 	Number of iterations we give the model "for free" 
						before its prediction error starts affecting loss. 
	"""

	loss = 0.0 
	with tf.GradientTape() as tape:
		for i in range(n_iters):
			reset = i==0
			cur_loss = model(video, reset_latent=reset)
			if i >= blind_iters:
				loss += cur_loss

	grads = tape.gradient(loss, model.trainable_weights)
	print("Gradient device: ", grads[0].device)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	return loss

	
	






"""
def train_model(model, dataset, optimizer, dset_size, num_epochs=1, iterations_per_batch=33, 
		in_window_size=100, test_window_size=500):

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
				
"""