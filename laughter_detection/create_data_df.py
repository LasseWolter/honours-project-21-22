from transcript_parsing import parse
import numpy as np
import preprocess as prep
import utils
import portion as P
import config as cfg
import pandas as pd
import os
import subprocess

DATA_DFS_DIR = 'data/data_dfs' 

# Taken from lhotse icsi recipe to minimise speaker overlap
PARTITIONS = {
    'train': [
        "Bdb001", "Bed002", "Bed003", "Bed004", "Bed005", "Bed006", "Bed008", "Bed009",
        "Bed010", "Bed011", "Bed012", "Bed013", "Bed014", "Bed015", "Bed016", "Bed017",
        "Bmr001", "Bmr002", "Bmr003", "Bmr005", "Bmr006", "Bmr007", "Bmr008", "Bmr009",
        "Bmr010", "Bmr011", "Bmr012", "Bmr014", "Bmr015", "Bmr016", "Bmr019", "Bmr020",
        "Bmr022", "Bmr023", "Bmr024", "Bmr025", "Bmr026", "Bmr027", "Bmr028", "Bmr029",
        "Bmr030", "Bmr031", "Bns002", "Bns003", "Bro003", "Bro004", "Bro005", "Bro007",
        "Bro008", "Bro010", "Bro011", "Bro012", "Bro013", "Bro014", "Bro015", "Bro016",
        "Bro017", "Bro018", "Bro019", "Bro022", "Bro023", "Bro024", "Bro025", "Bro026",
        "Bro027", "Bro028", "Bsr001", "Btr001", "Btr002", "Buw001",
    ],
    'dev': ["Bmr021", "Bns001"],
    'test': ["Bmr013", "Bmr018", "Bro021"]
}


def get_random_speech_segment(duration, meeting_id):
    '''
    Get a random speech segment from any channel in the passed meeting
    If there is an overlap between this segment and laughter/invalid regions, resample
    '''
    # Only consider segments with passed meeting_id
    info_df = parse.info_df[parse.info_df.meeting_id == meeting_id]
    # Get segment info for this segment from info_df
    num_of_rows = info_df.shape[0]
    row_ind = np.random.randint(0, num_of_rows)
    sample_seg = info_df.iloc[row_ind]
    start = np.random.uniform(0, sample_seg.length-duration)
    speech_seg = P.closed(utils.to_frames(
        start), utils.to_frames(start+duration))

    # If segment overlaps with any laughter or invalid segment, resample
    if (utils.seg_overlaps(speech_seg, [prep.laugh_index, prep.invalid_index], sample_seg.meeting_id, sample_seg.part_id)):
        return get_random_speech_segment(duration, meeting_id)
    else:
        sub_start, sub_duration = get_subsample(
            start, duration, cfg.train['subsample_duration'])
        return [start, duration, sub_start, sub_duration, sample_seg.path, 0]


def get_subsample(start, duration, subsample_duration):
    '''
    Take a segment defined by (start, duration) and return a subsample of passed duration within that region
    '''
    subsample_start = np.random.uniform(
        start, start+duration-subsample_duration)
    return subsample_start, subsample_duration


def create_data_df(data_dir):
    '''
    Create 3 dataframes (train,dev,test) with data exactly structured like in the model by Gillick et al.
    Columns:
        [region start, region duration, subsampled region start, subsampled region duration, audio path, label]

    Subsampled region are sampled once during creation. Later either the sampled values can be used or resampling can happen.
    (see Gillick et al. for more details)
    Duration of the subsamples is defined in config.py
    '''
    np.random.seed(cfg.train['random_seed'])
    speech_seg_lists = {'train': [], 'dev': [], 'test': []}
    laugh_seg_lists = {'train': [], 'dev': [], 'test': []}

    meeting_groups = parse.laugh_only_df.groupby('meeting_id')

    for meeting_id, meeting_laugh_df in meeting_groups:
        split = 'train'
        if meeting_id in PARTITIONS['dev']:
            split = 'dev'
        elif meeting_id in PARTITIONS['test']:
            split = 'test'

        # For each laughter segment get a random speech segment with the same length
        for _, laugh_seg in meeting_laugh_df.iterrows():
            # Get and append random speech segment of same length as current laugh segment
            speech_seg_lists[split].append(
                get_random_speech_segment(laugh_seg.length, meeting_id))

            # Subsample laugh segment and append to list
            audio_path = os.path.join(
                laugh_seg.meeting_id, f'{laugh_seg.chan}.sph')
            sub_start, sub_duration = get_subsample(
                laugh_seg.start, laugh_seg.length, cfg.train['subsample_duration'])

            laugh_seg_lists[split].append(
                [laugh_seg.start, laugh_seg.length, sub_start, sub_duration, audio_path, 1])

    # Columns for data_dfs - same for speech and laughter as they will be combined to one df
    cols = ['start', 'duration', 'sub_start',
            'sub_duration', 'audio_path', 'label']

    # Create output directory for dataframes
    subprocess.run(['mkdir', '-p', data_dir])

    for split in PARTITIONS.keys():  # [train,,test]
        speech_df = pd.DataFrame(speech_seg_lists[split], columns=cols)
        laugh_df = pd.DataFrame(laugh_seg_lists[split], columns=cols)
        whole_df = pd.concat([speech_df, laugh_df], ignore_index=True)
        # Round all floats to certain number of decimals (defined in config)
        whole_df = whole_df.round(cfg.train['float_decimals'])
        whole_df.to_csv(os.path.join(data_dir, f'{split}_df.csv'))

        # Check that df only contains correct meeting ids for this split
        audio_paths = whole_df.audio_path.unique().tolist()
        meeting_ids = set(map(lambda x: x.split('/')[0], audio_paths))
        mismatched_meetings = meeting_ids - set(PARTITIONS[split])
        assert len(
            mismatched_meetings) == 0, f"Found meetings in {split}_df with meeting_id not corresponding to that split, namely: {mismatched_meetings}"


if __name__ == "__main__":
    create_data_df(DATA_DFS_DIR)
