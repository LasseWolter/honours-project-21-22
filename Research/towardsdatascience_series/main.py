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
from classifier import AudioClassifier
import torch.nn as nn


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


def split_data():
    ''' Split data into training and testing data'''

    df = prepare_data()
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
    return (train_dl, val_dl)


def training(model, train_dl, num_epochs, device):
    ''' Training Loop'''
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(
                                                        len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            new_corrects = (prediction == labels).sum().item()
            correct_prediction += new_corrects
            total_prediction += prediction.shape[0]

            if i % 15 == 0:    # print every 10 mini-batches
                print('[%d, %5d] \nP:%s \nT:%s \nCorrect Preds: %d \nloss: %.3f' % (
                    epoch + 1, i + 1, prediction, labels, new_corrects, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('Finished Training')


def main():
    # Split data
    train_dl, val_dl = split_data()

    # Create the model and put it on the GPU if available
    myModel = AudioClassifier()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(device)
    # Check that it is on Cuda
    next(myModel.parameters()).device

    # Training
    num_epochs = 20   # Just for demo, adjust this higher.
    training(myModel, train_dl, num_epochs, device)
