from transcript_parsing import parse
import numpy as np
import preprocess as prep
import utils
import portion as P
import config as cfg
import pandas as pd
import os


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


def create_train_df():
    '''
    Create a dataframe with training data exactly structured like in the model by Gillick et al.
    Columns:
        [region start, region duration, subsampled region start, subsampled region duration, audio path, label]

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
    train_df = pd.concat([speech_df, laugh_df], ignore_index=True)
    print(train_df)


if __name__ == "__main__":
    create_train_df()
