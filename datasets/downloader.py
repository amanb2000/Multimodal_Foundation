"""
Script to download subsets of the HowTo100M dataset. 
"""
import os 
import sys 
import pdb

from yt_dlp import YoutubeDL
import pandas as pd 
import argparse
import tqdm 

## Parsing the arguments 
parser = argparse.ArgumentParser(
		description="Download some of the HowTo100M dataset.")

parser.add_argument('--csv-path', action='store', type=str, 
		help='path to the CSV with all the URLs in it', 
		default='HowTo100M_v1.csv')

parser.add_argument('--out-dir', action='store', type=str, 
		help='output folder for downloads, check for existing files.', 
		default = 'downloads')

parser.add_argument('--num-videos', action='store', type=int, 
		help="Number of videos to download from the source CSV. Selected randomly.", 
		default= "10")

parser.add_argument('--no-sub', action='store', type=bool, 
		help="Include this to refrain from downloading subtitles.")



args = parser.parse_args()
print("ARGS: \n", args)
print("\n")

if not os.path.exists(args.out_dir):
	os.makedirs(args.out_dir)

df = pd.read_csv(args.csv_path)

df = df.sample(n=args.num_videos)

print("After sampling: ")
print(df)

URLS = df['video_id'].tolist()

# ydl_opts = {'outtmpl': '%(id)s.%(ext)s', 
# 		'format-sort': 'vext=mp4',
# 		'format': 'best[height<=360]', 
# 		'output': args.out_dir, 
# 		'write-subs': True}

ydl_opts = {'write-subs': True}

print("OPTIONS: ", ydl_opts)

with YoutubeDL(ydl_opts) as ydl:
    ydl.download(URLS)
