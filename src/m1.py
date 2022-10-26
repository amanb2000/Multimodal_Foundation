""" # Model 1

This file contains code for the first full perceiver model. It was prototyped in
notebook `08_Perceiver_AE.ipynb`. 

This file only contains the `tf.keras.Model` definition. Data management, 
spacetime codes, etc. are done elsewhere. 
"""

## Import Box 
import random
import pathlib
import itertools
import collections
import math
import pdb

import tensorflow as tf 
from tensorflow import keras
import numpy as np
from tqdm import tqdm

# Custom modules
import video_loader as vl
import video_preprocess as vp 


## Custom layer: transformer layer. 
class TFLayer(keras.layers.Layer):
	def __init__(self, output_token_dim=None, n_heads=15, key_dim=15, mha_dropout=0.0): 
		""" Transformer block. Input -> MHA -> residual, layernorm -> 
				FFN -> residual (if possible). 

		The residual connections can only sometimes work. If the MHA step produces 
		tokens of a dimensionality DIFFERENT than the dimensionality of the 
		query tokens, we can't perform the residual step. 

		kwargs: 
			`output_dim`: 	Dimensionality of the output tokens. 
							If unspecified, defaults to the input token dimensionality
							(set during the `build()` function on first layer call). 
							The output will generally have the size [num_tokens, output_dim].
			`n_heads`:		Number of heads in the MHA layer. 
			`key_dim`:		Dimensionality of keys within each MHA layer. 
			`mha_dropout`: 	Dropout rate for the multihead attention layer. 
		"""
		super(TFLayer, self).__init__() 

		self.output_token_dim = output_token_dim

		# whether we perform the final residual; updated in `build() based on whether the output token dimensionality 
		self.ffn_residual = True 
		self.input_token_dim = None # set during `build()`

		self.MHA = keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim, dropout=mha_dropout)

		# We wait until `build` to construct our convolution layers.

		# Layernorms
		self.layer_norm_mha = tf.keras.layers.LayerNormalization()
		self.layer_norm_ffn = tf.keras.layers.LayerNormalization()

	def build(self, call_shapes): 
		self.MHA._build_from_signature(*call_shapes)
		q_shape, k_shape = call_shapes

		input_dim = q_shape[-1]
		self.input_token_dim = input_dim

		self.conv1 = keras.layers.Conv1D(input_dim*4, 1, activation="relu")

		# if not specified, the output tokens will be the same size as the input token size.
		if self.output_token_dim == None:
			self.output_token_dim = input_dim 
		
		if not (self.output_token_dim == input_dim): 
			self.ffn_residual = False

		self.conv2 = keras.layers.Conv1D(self.output_token_dim, 1, activation="linear") 

	def call(self, kv_list, final_layernorm=True, verbose=False): 
		""" Invokes the transformer block on the given queries and keys. 
		"""
		queries, keys = kv_list

		# Multihead attention sub-block.
		mha_out = self.MHA(queries, keys)
		mha_out = self.layer_norm_mha(mha_out + queries) # residual + layernorm

		# FFN sub-block
		ffn1_out = self.conv1(mha_out)
		ffn2_out = self.conv2(ffn1_out)
		if self.ffn_residual:
			ffn2_out += mha_out # performing residual if the out_dim == in_dim

		if final_layernorm:
			return self.layer_norm_ffn(ffn2_out) # final layernorm
		return ffn2_out



## Custom layer: encoder layer. 
class PAE_Encoder(keras.layers.Layer):
	def __init__(self, n_blocks, p_droptoken=0.5, re_droptoken=True, 
			tfblock_residual=True, n_heads=15, key_dim=15, mha_dropout=0.0): 
		""" 		
		Perceiver AE encoder layer. Queries during the call should be the 
		current `latent` state, and the key-values should be the incoming 
		byte array.

		The encoder sequentially re-queries the incoming byte array using the 
		latent state. The latent state used for re-querying is summed with 
		the result of each block, then the layernorm is taken (i.e., residual 
		+ layernorm). 

		args: 
			`n_heads`, `key_dim`, `mha_dropout`: Params uniformly applied to all transformer 
							blocks. 
			`n_blocks`: 	Number transformer blocks performing successive 
							requerying of the input. 
			`p_droptoken`: 	Portion of tokens to MAINTAIN from the input. 
			`re_droptoken`: Do we reselect the dropped tokens every time we 
							successively re-query the input? 
		kwargs: 
			`output_dim`: 	Dimensionality of the output tokens. 
		"""
		super(PAE_Encoder, self).__init__() 

		# Recording parameters. 
		self.n_blocks = n_blocks
		self.p_droptoken = p_droptoken
		self.re_droptoken = re_droptoken
		self.tfblock_residual = tfblock_residual

		self.n_heads = n_heads
		self.key_dim = key_dim
		self.mha_dropout = mha_dropout

		# Component layers
		self.tf_layers	 = [TFLayer(n_heads=self.n_heads, key_dim=self.key_dim, mha_dropout=self.mha_dropout) for i in range(self.n_blocks)]
		self.layer_norms = [tf.keras.layers.LayerNormalization() for i in range(self.n_blocks)]


	def call(self, kv_list, no_drop=False,  verbose=False): 
		""" Invokes the encoder module on a latent state and an input array. 

		args: 
			`kv_list`: 		Tuple or list [current_latent, input_array]
							current_latent.shape = [batch_size, N, C]
							input_array.shape    = [batch_size, M, D]
		"""
		latent, input_byte_array = kv_list

		# Length = # tokens in input
		if not no_drop:
			droptoken_mask = tf.random.uniform([input_byte_array.shape[1]]) < self.p_droptoken
			current_input = tf.boolean_mask(input_byte_array, droptoken_mask, axis=1)
		else: 
			current_input = input_byte_array

		# iterating through: 
		for i in range(self.n_blocks):
			assert current_input.shape[1] > 0, "Random sampling failed!"

			latent_ = self.tf_layers[i]([latent, current_input])

			# If we are applying tfblock_residuals: 
			if self.tfblock_residual: 
				latent_ = self.layer_norms[i](latent_ + latent)

			latent = latent_

			# If we are going to reselect tokens from the input for every 
			# re-querying: 
			if self.re_droptoken and not no_drop: 
				droptoken_mask = tf.random.uniform([input_byte_array.shape[1]]) 
				current_input = tf.boolean_mask(input_byte_array, droptoken_mask, axis=1)

		return latent


## Custom layer: decoder layer. 
class PAE_Decoder(keras.layers.Layer):
	def __init__(self, output_patch_dim, n_blocks, expansion_block_num, 
			tfblock_residual=True, n_heads=15, key_dim=15, mha_dropout=0.0): 
		""" Perceiver AE decoder layer. This layer maps from a pair of tensors: 
			query		= spacetime codes of the patches we want to reconstruct, 
			key-values	= current latent tensor.

		Much like the encoder layer, several transformer blocks are used to 
		successively re-query the key-value tensor (in this case, the latent 
		tensor from the model). It's like making "successive approximations" of 
		the reconstruction. 

		The query is the spacetime codes of the patches we want to reconstruct.
		This query is iteratively transformed into the reconstruction itself. 
		

		args: 
			`output_patch_dim`: Dimensionality of the patches we want to 
						reconstruct. 
			`n_blocks`: Number transformer blocks performing successive 
						requerying of the input. 

			`expansion_block_num`: On which of the `n_blocks` blocks do we make 
						the transition (on the query) from the token
						dimensionality of the spacetime codes to the token
						dimensionality of the patches we want to reconstruct?
			
		kwargs: 
			`tfblock_residual`: Do we include residual connections around the 
						transformer blocks? 

			`n_heads`, `key_dim`, `mha_dropout`: Params uniformly applied to all 
						transformer blocks. 

			`output_dim`: 	Dimensionality of the output tokens. 
		"""
		super(PAE_Decoder, self).__init__() 

		# Recording parameters. 
		self.output_patch_dim = output_patch_dim
		self.n_blocks = n_blocks
		self.expansion_block_num = expansion_block_num
		self.tfblock_residual = tfblock_residual

		self.n_heads = n_heads
		self.key_dim = key_dim
		self.mha_dropout = mha_dropout

		# Component layers
		self.tf_layers = []

		for i in range(self.n_blocks):
			add_me = None
			if i == self.expansion_block_num:
				add_me = TFLayer(output_token_dim=self.output_patch_dim, n_heads=self.n_heads, key_dim=self.key_dim, mha_dropout=self.mha_dropout)
			else:
				add_me = TFLayer(n_heads=self.n_heads, key_dim=self.key_dim, mha_dropout=self.mha_dropout)

			self.tf_layers.append(add_me)

		self.layer_norms = [tf.keras.layers.LayerNormalization() for i in range(self.n_blocks)]


	def call(self, kv_list, verbose=False): 
		""" Invokes the encoder module on a latent state and an input array. 

		args: 
			`kv_list`: 		Tuple or list [spacetime_codes, latents]
							current_latent.shape = [batch_size, N, C]
							input_array.shape    = [batch_size, M, D]
		"""
		# pdb.set_trace()
		query_spacetime_code, latents = kv_list

		reconstruction = query_spacetime_code 

		# iterating through: 
		for i in range(self.n_blocks):
			if i != self.n_blocks-1:
				reconstruction_ = self.tf_layers[i]([reconstruction, latents])
			else:
				reconstruction_ = self.tf_layers[i]([reconstruction, latents], final_layernorm=False)


			# If we are applying tfblock_residuals: 
			if self.tfblock_residual and i != self.expansion_block_num and i != self.n_blocks-1: 
				reconstruction_ = self.layer_norms[i](reconstruction_ + reconstruction)

			reconstruction = reconstruction_ 

		return reconstruction
				


## Custom layer: Latent-latent evolver 
class PAE_Latent_Evolver(keras.layers.Layer):
	def __init__(self, n_blocks, distinct_blocks=True, 
			tfblock_residual=True, n_heads=15, key_dim=15, mha_dropout=0.0): 
		""" Perceiver AE latent evolver layer. 


		args: 
			`n_blocks`: 	Total number of transformer blocks the data goes 
							through before being returned. NOT necessarily 
							distinct weights!
			
		kwargs: 
			`distrinct_blocks`: Should all the blocks share weights?

			`tfblock_residual`: Do we include residual connections around the 
						transformer blocks? 

			`n_heads`, `key_dim`, `mha_dropout`: Params uniformly applied to all 
						transformer blocks. 
		"""
		super(PAE_Latent_Evolver, self).__init__() 

		# Recording parameters
		self.n_blocks 		 = n_blocks
		self.distinct_blocks = distinct_blocks
		self.tfblock_residual = tfblock_residual

		self.n_heads = n_heads
		self.key_dim = key_dim
		self.mha_dropout = mha_dropout

		# Creating the layers 
		if self.distinct_blocks: 
			self.tf_layers = [TFLayer(n_heads=self.n_heads, key_dim=self.key_dim, mha_dropout=self.mha_dropout) for i in range(self.n_blocks)]
		else: 
			self.tf_layers = [TFLayer(n_heads=self.n_heads, key_dim=self.key_dim, mha_dropout=self.mha_dropout)]

		self.layer_norm = tf.keras.layers.LayerNormalization()

	def call(self, latent_in):
		latent = latent_in

		for i in range(self.n_blocks):
			j = i if self.distinct_blocks else 0 

			latent_ = self.tf_layers[j]([latent, latent])

			if self.tfblock_residual: 
				latent_ = latent + latent_
				latent_ = self.layer_norm(latent_)
			
			latent = latent_

		return latent
		

## The model itself

class PerceiverAE(keras.Model):
	def __init__(self, loss_fn, encoder_module, latent_module, decoder_module, 
			latent_dims=(90, 77), code_dim=191): 
		""" Full perceiver AE model. 

		args: 
			`loss_fn`: 		Keras-style loss function. Applied to calculate 
							reconstruction loss. 

			`encoder_module`: Instance of PAE_Encoder class (see above).

			`latent_module`: Instance of the PAE_Latent_Evolver class (see 
							above).

			`decoder_module`: Instance of the PAE_Decoder class (see above).

		kwargs: 
			`latent_dims`: 	[num_tokens, dim_tokens] in the latent tensor. 

			`code_dim`: 	Dimensionality of the spacetime Fourier codes used
							in the input byte array tensor.
		"""
		super(PerceiverAE, self).__init__()

		self.N, self.C = latent_dims  	# N rows (tokens), each with dimensionality C.
		self.loss_fn = loss_fn

		# 3 primary submodules
		self.encoder = encoder_module
		self.latent_ev = latent_module
		self.decoder = decoder_module

		self.code_dim = code_dim

		# Latent state management:
		# The latent tensor has shape [batch, N, C] -- that way we can have 
		# different latent states for each batch dimension. 
		# `source_latent` is the initial value for each latent across 
		# batches. 

		# In the future we will make `source_latent` a learnable parameter... 
		self.source_latent = tf.random.normal([1, self.N, self.C])
		self.latent = None

	def reset_latent(self, B=1): 
		self.latent = tf.concat([self.source_latent for i in range(B)], axis=0)

	def call(self, reconstruct_me, reset_latent=False, return_prediction=False):
		if reset_latent or self.latent == None: 
			B = reconstruct_me.shape[0]
			self.reset_latent(B=B)

		# Calculate loss on trying to predict the `reconstruct_me`: 
		spacetime_codes = reconstruct_me[:,:,-self.code_dim:]
		prediction = self.decoder([spacetime_codes, self.latent])
		surprise = self.loss_fn(prediction, reconstruct_me[:,:,:-self.code_dim])

		# Incorporating new information into the latent 
		self.latent = self.encoder([self.latent, reconstruct_me])

		# Evolving the latent state autonomously
		self.latent = self.latent_ev(self.latent)

		# Returning the surprise
		if return_prediction: 
			return surprise, prediction

		return surprise




