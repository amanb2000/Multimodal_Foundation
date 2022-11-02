""" 
Benchmarking different dataset management methods.
"""
import os 
import tensorflow as tf
import video_loader as vl
import video_preprocess as vp
import pdb
import numpy as np

print("\n\n\n")

DATA_FOLDER = "../datasets/downloads"
mp4_list = os.listdir(DATA_FOLDER)
print("MP4 LIST: ", mp4_list[:10])

def load_video_file(fname): 
	assert fname==1, f"{fname}"
	try:
		return tf.convert_to_tensor(vl.get_single_video_tensor(fname, 300, output_size=(240,360)))
	except:
		return tf.zeros([300,240,360,3])

train_dataset = tf.data.Dataset.list_files(os.path.join(DATA_FOLDER, "*.mp4"))

for ele in train_dataset: 
	print("\n\n\nELEMENT: ",ele)
	break


videoset = train_dataset.map(lambda x: tf.py_function(load_video_file, [x], [tf.float32]))

num_frames = 300
output_size = (240,360)

# videoset = textset.map(lambda x: vl.get_single_video_tensor(os.path.join(DATA_FOLDER, x.numpy()), num_frames, output_size=output_size))

cnt=0
for ele in videoset: 
	print(f"{cnt} -- ", tf.norm(ele).numpy())
