"""
    This script creates audio files for each laughter event
    in a given .TextGrid file and stores them in an output directory.
    The .sph file corresponding to the .TextGrid file needs to be passed as well.
    These are laughter events from one specific channel, NOT the whole meeting
"""
import os
import sys
import subprocess
import textgrids

if len(sys.argv) < 4:
    print("Usage laughs_to_wav.py <textgrid-file> <sph-file> <out-dir>")
    sys.exit()

textgrid_file = sys.argv[1]
sph_file = sys.argv[2]
sph_base_file = os.path.basename(sph_file)
out_dir = sys.argv[3]
if not os.path.isdir(out_dir):
    subprocess.run(['mkdir','-p',out_dir])

laughs = []
grid = textgrids.TextGrid(textgrid_file)
for interval in grid['laughter']:
    if str(interval.text) == 'laugh':
        laughs.append((interval.xmin, interval.xmax))

for ind, laugh in enumerate(laughs):
    subprocess.run(
        ["sph2pipe", '-t', f'{laugh[0]}:{laugh[1]}', sph_file, f'{out_dir}/{sph_base_file}_{ind}.wav'])

# Concat laughs to one stream with break.wav as 'delimiter'
# Assumes that corresponding bash script is in the same folder
subprocess.run(['./concat_laughs.sh', out_dir])

