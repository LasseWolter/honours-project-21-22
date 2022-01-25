from torch.utils.data import DataLoader
from lhotse import CutSet, Fbank, MonoCut, LilcomFilesWriter
from lhotse.dataset import VadDataset, SingleCutSampler
from lhotse.recipes import prepare_icsi
import pandas as pd

# Prepare data manifests from a raw corpus distribution.
# The RecordingSet describes the metadata about audio recordings;
# the sampling rate, number of channels, duration, etc.
# The SupervisionSet describes metadata about supervision segments:
# the transcript, speaker, language, and so on.
icsi, chan_idx_map = prepare_icsi(
    audio_dir='data/speech/test', transcripts_dir='data/transcripts/test', output_dir='lhotse_manifests')


# Read data_dfs containing the samples for train,val,test split
# train_df = pd.read_csv('data_dfs/train_df.csv')
# val_df = pd.read_csv('data_dfs/val_df.csv')
# test_df = pd.read_csv('data_dfs/test_df.csv')

dummy_df = pd.read_csv('data_dfs/dummy_df.csv')


# CutSet is the workhorse of Lhotse, allowing for flexible data manipulation.
# We create 5-second cuts by traversing ICSI recordings in windows.
# No audio data is actually loaded into memory or stored to disk at this point.
# Columns of dataframes look like this:
#   cols = ['start', 'duration', 'sub_start', 'sub_duration', 'audio_path', 'label']
cut_list = []
for ind, row in dummy_df.iterrows():
    meeting_id = row.audio_path.split('/')[0]
    channel = row.audio_path.split('/')[1].split('.')[0]
    chan_id = chan_idx_map[meeting_id][channel]
    # Just for now, because not all channels have been downloaded -> mistake in prepare_icsi()
    if chan_id > 6:
        continue
    # Dummy recording is in the train split of the lhotse manifest
    cut = MonoCut(id=f'dummy_{ind}', start=row.start, duration=row.duration,
                  recording=icsi['train']['recordings'][meeting_id], channel=chan_id)
    cut_list.append(cut)

cutset = CutSet.from_cuts(cut_list)

start_set = cutset.subset(first=10)

cuts = cutset.compute_and_store_features(
    extractor=Fbank(),
    storage_path='feats',
    num_jobs=8,
    storage_type=LilcomFilesWriter
)

# Construct a Pytorch Dataset class for Voice Activity Detection task:
dataset = VadDataset(cuts)
sampler = SingleCutSampler(cuts)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=None)
batch = next(iter(dataloader))
