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

args = parser.parse_args()
print("ARGS: \n\t", args)
print("\n")

out_dir = args.out_dir 
sub_dir = os.path.join(out_dir, 'subs')

langs = 'en,zh-Hans,hi,es,fr,jp'




## Dispatch command 
def download_url(url, opts):
	# url is a string of the video id; opts is a dictionary with all the 
	# commandline args for `yt-dlp` call. 

