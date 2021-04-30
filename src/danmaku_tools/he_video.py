import json
import os
import sys

import ffmpeg
from ffmpeg_smart_trim.trim import TrimVideo


# parser = argparse.ArgumentParser(description='Cut HE range in BiliBili live recordings')
# parser.add_argument('--he_range', type=str, help='path to he range file')
# parser.add_argument('--video', type=str, help='path to video to cut')
#
# args = parser.parse_args()
class args:
    name = "128308-20210427-224046"
    he_range = f"/Users/jackie/Downloads/{name}.all.he_range.txt"
    video = f"/Users/jackie/Downloads/{name}.all.bar.mp4"
    output = f"/Users/jackie/Downloads/{name}.he.all.bar.mp4"


# %%
with open(args.he_range, "r") as he_file:
    he_range = json.load(he_file)

if len(he_range) == 0:
    sys.exit(1)

EXPAND_TIME = 10
expanded_he_range = [(current_range[0] - EXPAND_TIME, current_range[1] + EXPAND_TIME) for current_range in he_range]

he_range = []
last_range = None

for current_range in expanded_he_range:
    if last_range is None:
        last_range = current_range
        continue
    else:
        if last_range[1] >= current_range[0]:
            last_range = (last_range[0], current_range[1])
        else:
            he_range += [last_range]
            last_range = current_range

# noinspection PyUnboundLocalVariable
he_range += [current_range]

# %%
print("Parsing video file...")
video = TrimVideo(args.video, time_range=(he_range[0][0], he_range[-1][1]))

files, fast_trims, slow_trims = [], [], []

for i, current_he_range in enumerate(he_range):
    current_files, current_fast_trims, current_slow_trims \
        = video.generate_trim(current_he_range[0], current_he_range[1], prefix=str(i))
    files += current_files
    fast_trims += current_fast_trims
    slow_trims += current_slow_trims
print("Trimming video file...")
if len(fast_trims) > 0:
    print(ffmpeg.merge_outputs(*fast_trims).compile())
    ffmpeg.merge_outputs(*fast_trims).run(overwrite_output=True)
if len(slow_trims) > 0:
    print(ffmpeg.merge_outputs(*slow_trims).compile())
    ffmpeg.merge_outputs(*slow_trims).run(overwrite_output=True)
temp_merge_path = os.path.join(video.temp_dir, "merged.mp4")
merge_cmd = video.generate_merge(files, temp_merge_path)
print(merge_cmd.compile())
print("Merging video file...")
merge_cmd.run(overwrite_output=True)
merged_input = ffmpeg.input(temp_merge_path)
copy_cmd = ffmpeg.output(merged_input, args.output, c='copy')
print("Copying video to destination...")
print(copy_cmd.compile())
copy_cmd.run(overwrite_output=True)
video.clean_temp()
