"""
This download script dispatches the `yt-dlp` jobs to the commandline. 

REQUIREMENTS: 
 - ~360p resolution 
 - ~100M maximum size (captcha acts up otherwise).
 - Subtitles are a must. 

COMMAND EXAMPLE: 

	yt-dlp --write-auto-subs -o 'downloads/%(id)s.%(ext)s' -o 'subtitle:downloads/subs/%(id)s.%(ext)s' --format-sort vext --format 'best[height<=360]' --sub-langs en,zh-Hans,hi,es,fr,jp  --max-filesize 100M gIURSGIXX0Y
"""

## Imports
import subprocess 
import os
import sys
import re
import argparse
import pdb
from concurrent.futures import ProcessPoolExecutor
import time 
import random

import pandas as pd


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

parser.add_argument('--max-filesize', action='store', type=str, help="Max filesize (e.g., 100M).", 
		default="100M")

args = parser.parse_args()
print("ARGS: \n\t", args)
print("\n")

out_dir = args.out_dir 
sub_dir = os.path.join(out_dir, 'subs')

langs = 'en,zh-Hans,hi,es,fr,jp'

## Name templates 
template = "%(id)s.%(ext)s"
vid_template = os.path.join(out_dir, template) 
sub_template = os.path.join('subtitle:'+out_dir, 'subs', template)

## Command manufacturing
base_command = ["./yt-dlp", "--write-auto-subs", "--format-sort", "vext", "--format", "best[height<=360]"]
base_command += ["--sub-langs", langs] 
base_command += ["-o", vid_template]
base_command += ["-o", sub_template] 
base_command += ["--max-filesize", args.max_filesize]

print("\nBASE COMMAND: \t", base_command)
print("")


## Getting dataframe 
df = pd.read_csv(args.csv_path)
df = df.sample(n=args.num_videos) #sampling

URLS = df['video_id'].tolist()

print("DATAFRAME: \n", df)
print("URLS: ", URLS)



## Dispatch command 
def download_url(url, cmd_base, base='downloads'):
	# url is a string of the video id; opts is a dictionary with all the 
	# commandline args for `yt-dlp` call. 
	full_command = cmd_base + [url]
	# print("\t", full_command)

	print(f"Downloading {url}...")

	if os.path.exists(os.path.join(base, url+".mp4")):
		print(f"{url}.mp4 already exists in {base}! Skipping.")

	x = subprocess.run(full_command, capture_output=True, text=True)
	# print(x)
	if x.returncode == 0: 
		print(f"Done downloading {url}!\n")
	else: 
		print(f"ERROR WHEN DOWNLOADING {url}!!!")

	return x


num_naps = 0
cnt = 0
streak = 0
good_streak = 0
print("\n\n\nSTARTING DOWNLOADS!")
for u in URLS: 
	cnt+=1
	retval = download_url(u, base_command, base=out_dir)

	if retval != None and retval.returncode != 0: 
		num_naps+=1
		streak += 1
		good_streak = 0

		print(retval)
		print(f"\n\nNap #{num_naps} of {cnt} tries, {streak} streak...")
		time.sleep(30)
		print("Wakeup!\n")
		# exit(0)
	else: 
		streak = 0
		good_streak += 1
		print(f"\tSuccess streak: {good_streak} \tTotal: {cnt}\n")

	time.sleep(5)
