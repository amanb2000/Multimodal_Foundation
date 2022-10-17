# Scratch Notebooks 

 - `01_HowTo100M_Exploration.ipynb`: Prototyping the dataset wrangling script. 
 - `02_Video_Loading.ipynb`: Video dataset prototyping script. 
 - `03_Transformer_Preliminaries.ipynb`: Examples/proof-of-concepts for unfamiliar aspects of the transformer architecture. 
	 - Spacetime patch generation prototype. 
	 - Fourier positional spacetime codes prototype. 
	 - Playing with MHA blocks. 
 - `04_Overfit_TAE.ipynb`: Very simple transformer autoencoder -- didn't really work. 
 - `05_Attention_Play.ipynb`: Getting a feel for how MHA blocks and FFN blocks work, making sure I understand the API. 
 - `06_TAE_Full.ipynb`: Implement a simple, full version of the transformer autoencoder. 
	 - **Encoder**: Single MHA -> FFN block with query = latent matrix, key-values = input matrix. 
	 - **Latent-latent**: Adjustable # repeats and # transformer blocks (each block is MHA -> FFN with layernorms).
	 - **Decoder**: Single MHA -> FFN block with query = positional codes for desired spacetime patches, key-values = latent matrix. 
	 - Successfully overfit to 16 black-and-white images where tokens = image rows. Every iteration, the entire image was fed in and the entire image was decoded. Literally a transformer autoencoder. 
 - `07_Untokenize.ipynb`: Prototyping the unpatching routine for spacetime patches. Implemented in `src/video_preproccess.py`. 

