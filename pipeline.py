import os

from ile_detection import *
from infering import *
from summary import *

FIND_START = True
INFER = True
SUMMARY = True

root_dir = ''

videos = []

for root, dirs, files in os.walk(root_dir):
    mp4lst = []
    for file in files:
        if file.endswith('.mp4'):
            videos.append(f'{root.split("/")[-1]}-{file.split(".")[0]}')
            break

if __name__ == "__main__":
    for itr, video in enumerate(videos):
        print(f"Processing {video}...({itr+1}/{len(videos)})")
        try:
        
            print("Finding Start Points...")
            print(ile_dect(video, root_dir))

            print("Localizing and Scoring...")
            st_time = pd.read_csv('./documents/start_time.csv').set_index('video').loc[video]['st_time']
            print(infer(video, st_time, root_dir))

            print("Summarizing...")
            print(summary(video, root_dir))

            print()
        except:
            print(f'Error encountered when processing {video}, SKIP!\n')
    print("All Done!")
