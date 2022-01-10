from transcript_parsing import parse
import numpy as np
import preprocess as prep
import utils
import portion as P
import config as cfg
import pandas as pd
import os
import subprocess


def get_random_speech_segment(duration):
    # Get segment info for this segment from info_df
    row_number = parse.info_df.shape[0]
    row_ind = np.random.randint(0, row_number)
    sample_seg = parse.info_df.iloc[row_ind]
    start = np.random.uniform(0, sample_seg.length-duration)
    speech_seg = P.closed(utils.to_frames(
        start), utils.to_frames(start+duration))

    # If segment overlaps with any laughter or invalid segment resample
    if (utils.seg_overlaps(speech_seg, [prep.laugh_index, prep.invalid_index], sample_seg.meeting_id, sample_seg.part_id)):
        return get_random_speech_segment(duration)
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


def train_val_test_split(df, fracs):
    '''
    Split pd.Dataframe into 3 dataframes of given fractions [train, validation]. The test set takes the remainder fraction.
    '''
    train = df.sample(frac=fracs[0])
    remain_df = df.drop(index=train.index)
    # Recalculate fractions for remaining_df
    val_frac = 0.1 / (1-fracs[0])
    val = remain_df.sample(frac=val_frac)
    test = remain_df.drop(index=val.index)
    return train, val, test


def create_data_df(data_dir):
    '''
    Create a dataframe with training data exactly structured like in the model by Gillick et al.
    Columns:
        [region start, region duration, subsampled region start, subsampled region duration, audio path, label]

    Also creates 3 small splits for debuggin (all .csv files preceded with 'small_')

    Subsampled region are sampled once during creation. Later either the sampled values can be used or resampling can happen.
    (see Gillick et al. for more details)
    Duration of the subsamples is defined in config.py
    '''
    np.random.seed(cfg.train['random_seed'])
    speech_seg_list = []
    laugh_seg_list = []

    # For each laughter segment get a random speech segment with the same length
    for _, laugh_seg in parse.laugh_only_df.iterrows():
        # Get and append random speech segment of same length as current laugh segment
        speech_seg_list.append(get_random_speech_segment(laugh_seg.length))

        # Subsample laugh segment and append to list
        audio_path = os.path.join(
            laugh_seg.meeting_id, f'{laugh_seg.chan}.sph')
        sub_start, sub_duration = get_subsample(
            laugh_seg.start, laugh_seg.length, cfg.train['subsample_duration'])

        laugh_seg_list.append(
            [laugh_seg.start, laugh_seg.length, sub_start, sub_duration, audio_path, 1])

    cols = ['start', 'duration', 'sub_start',
            'sub_duration', 'audio_path', 'label']
    speech_df = pd.DataFrame(speech_seg_list, columns=cols)
    laugh_df = pd.DataFrame(laugh_seg_list, columns=cols)
    whole_df = pd.concat([speech_df, laugh_df], ignore_index=True)

    # Round all floats to certain number of decimals (defined in config)
    whole_df = whole_df.round(cfg.train['float_decimals'])
    small_df = whole_df[whole_df.audio_path.str.contains('Bdb001')]

    subprocess.run(['mkdir', '-p', data_dir])
    train_df, val_df, test_df = train_val_test_split(
        whole_df, cfg.train['train_val_test_split'])
    train_df.to_csv(os.path.join(data_dir, 'train_df.csv'))
    val_df.to_csv(os.path.join(data_dir, 'val_df.csv'))
    test_df.to_csv(os.path.join(data_dir, 'test_df.csv'))

    # Create small split for debugging
    small_train_df, small_val_df, small_test_df = train_val_test_split(
        small_df, cfg.train['train_val_test_split'])
    small_train_df.to_csv(os.path.join(data_dir, 'small_train_df.csv'))
    small_val_df.to_csv(os.path.join(data_dir, 'small_val_df.csv'))
    small_test_df.to_csv(os.path.join(data_dir, 'small_test_df.csv'))

    print(whole_df)


if __name__ == "__main__":
    create_data_df('data_dfs')
