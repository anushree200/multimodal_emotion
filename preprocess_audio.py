'''
groundwork:

spectogram is like a graph for cnn to process. 
x axis - time , y axis - frequency , intensity at each point - amplitude

neutral - medium frequency - medium amplitude
calm - low-medium - low
happy - high - medium-high
sad - low - low
angry - high - high
fearful - high - low-medium
diguist - low-medium - low-medium
surprised - high - high

'''

import pandas as pd
import numpy as np
import librosa
import cv2
import matplotlib.pyplot as plt

def audio_to_spectrogram(file_path, img_size=(128, 128)):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = cv2.resize(S_db, img_size, interpolation=cv2.INTER_AREA)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    return S_db

train_df = pd.read_csv('ravdess_train.csv')
test_df = pd.read_csv('ravdess_test.csv')

train_spectrograms = []
train_labels = []
for _, row in train_df.iterrows():
    spec = audio_to_spectrogram(row['file_path'])
    train_spectrograms.append(spec)
    train_labels.append(row['emotion'])
train_spectrograms = np.array(train_spectrograms)[..., np.newaxis]
train_labels = np.array(train_labels)

test_spectrograms = []
test_labels = []
for _, row in test_df.iterrows():
    spec = audio_to_spectrogram(row['file_path'])
    test_spectrograms.append(spec)
    test_labels.append(row['emotion'])
test_spectrograms = np.array(test_spectrograms)[..., np.newaxis]
test_labels = np.array(test_labels)

np.save('train_spectrograms.npy', train_spectrograms)
np.save('train_labels.npy', train_labels)
np.save('test_spectrograms.npy', test_spectrograms)
np.save('test_labels.npy', test_labels)


print("Preprocessing done! Saved spectrograms and labels.")