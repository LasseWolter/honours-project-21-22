'''
Most code taken from https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
'''
from torch.utils.data import random_split
from pathlib import Path
import pandas as pd
from soundDS import SoundDS
import matplotlib.pyplot as plt
import torch
import librosa.display


def prepare_data():
    ''' Prepare training data from Metadata file '''
    download_path = Path.cwd()/'data/UrbanSound8K'

    # Read metadata file
    metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    # Construct file path by concatenating fold and file name
    df['relative_path'] = '/fold' + \
        df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

    # Take relevant columns
    df = df[['relative_path', 'classID']]
    df.head()
    return df


def plot_specgram(spec_tensor, sample_rate):
    '''Plot spectrogram using librosa display'''
    specgram = spec_tensor.numpy()
    librosa.display.specshow(specgram, sr=sample_rate)
    plt.show(block=False)


def load_data():
    data_path = Path.cwd() / 'data/UrbanSound8K/audio'
    df = prepare_data()
    myds = SoundDS(df, data_path)
    return myds


def split():
    ''' Split data into training and testing data'''

    data_path = Path.cwd() / 'data/UrbanSound8K/audio'
    myds = SoundDS(df, data_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)


# Run prepare to have the df available in global context
df = prepare_data()
