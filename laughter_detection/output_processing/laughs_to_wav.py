"""
    This script creates audio files for each laughter event in:
    1)  a given .TextGrid file and stores them in an output directory.
        The .sph file corresponding to the .TextGrid file needs to be passed as well.
        These are laughter events from one specific channel, NOT the whole meeting
    2)  a give .csv file and a given meeting (set in the global variable below)
        no need to pass an .sph file as they are automatically parsed from the audio dir
    Finally it runs a bash script to combine all audio files with a short 'audio-delimiter' for 
    easier manual testing
        
"""
import os
import sys
import subprocess
import textgrids
import pandas as pd

MEETING = 'Buw001'

if len(sys.argv) < 3:
    print("Usage laughs_to_wav.py <input-file> <out-dir> <sph_file (for .TextGrid only)>")
    sys.exit()

input_file = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.isdir(out_dir):
    subprocess.run(['mkdir','-p',out_dir])

if input_file.endswith('.TextGrid'):
    sph_file = sys.argv[3]
    sph_base_file = os.path.basename(sph_file)
    laughs = []
    grid = textgrids.TextGrid(input_file)
    for interval in grid['laughter']:
        if str(interval.text) == 'laugh':
            laughs.append((interval.xmin, interval.xmax))

    for ind, laugh in enumerate(laughs):
        print(f'Generating wav for index {ind}...')
        subprocess.run(
            ["sph2pipe", '-t', f'{laugh[0]}:{laugh[1]}', sph_file, f'{out_dir}/{sph_base_file}_{ind}.wav'])

elif input_file.endswith('.csv'):
    df = pd.read_csv(input_file)
    df = df[df['Meeting'] == MEETING]
    df = df[['Channel','Start', 'End']]
    group = df.groupby('Channel')
    for chan, df in group:
        sph_file = f'./audio/Signals/{MEETING}/{chan}.sph' 
        sph_base_file = os.path.basename(sph_file)
        for ind, row in df.iterrows():
            start =row['Start']
            stop = row['End']
            print(f'Generating wav for channel {chan} index {ind}...')
            subprocess.run(
                ["sph2pipe", '-t', f'{start}:{stop}', sph_file, f'{out_dir}/{sph_base_file}_{ind}.wav'])


# Concat laughs to one stream with break.wav as 'delimiter'
# Assumes that corresponding bash script is in the same folder
subprocess.run(['./concat_laughs.sh', out_dir])

