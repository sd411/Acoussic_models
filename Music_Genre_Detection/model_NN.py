# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:36:28 2021

@author: dell
"""

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')




cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'genres_original/{g}'):
        try:
            songname = f'genres_original/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
            plt.clf()
        except:
            continue
        
#generating a csv file to store the features of the audio file
header = 'filename chroma_stft chroma_stft_var rmse_var rmse_mean spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var y_harm_mean y_harm_var y_perc_mean y_perc_var tempo'
for i in range(1, 21):
    header += f' mfcc_mean{i} mfcc_var{i}'
header += ' label'
header = header.split()
        
file = open('datalarger2.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    print(g)
    for filename in os.listdir(f'audio3sec/{g}'):
        try:
            print(filename)
            songname = f'audio3sec/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            rmse = librosa.feature.rms(y=y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            y_harm, y_perc = librosa.effects.hpss(y)
            tempo, _ = librosa.beat.beat_track(y, sr = sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.var(rmse)} {np.mean(rmse)} {np.mean(spec_cent)} {np.var(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)} {np.mean(y_harm)} {np.var(y_harm)} {np.mean(y_perc)} {np.var(y_perc)} {tempo}'    
            for e in mfcc:
                to_append += f' {np.mean(e)} {np.var(e)}'
            to_append += f' {g}'
            file = open('datalarger2.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
            print("Done")
        except:
            continue
        
        
        
#data preprocessing       
data = pd.read_csv('datalarger2.csv')
data.head()
data = data.sample(frac=1).reset_index(drop=True)        
data.shape
data = data.drop(['filename'],axis=1)
        
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

                        
#data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                        
                        
# Defining model
from keras import models
from keras import layers
from keras.layers import Dropout     
x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=30,
          batch_size=32,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)
        
predictions = model.predict(X_test)
        
#encoder.classes_
x = (np.argmax(predictions[0]))
x = np.array(x)
b = x.ravel()     
encoder.inverse_transform(b)

#helper functions
def get_scores(X,sc,model):
    x = np.array(X)
    x = x.reshape(1,-1)
    x = sc.transform(x)
    y = model.predict(x)
    y =y.flatten()
    y=y.tolist()
    #return y
    for item in range(len(y)):
        print(f'{encoder.classes_[item]} score is ' + f'{y[item]}')
        

def get_feature_vector(path):
    ls = []
    songname = path
    y, sr = librosa.load(songname, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    y_harm, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y, sr = sr)
    ls.append(np.mean(chroma_stft))
    ls.append(np.var(chroma_stft))
    ls.append(np.var(rmse)) 
    ls.append(np.mean(rmse)) 
    ls.append(np.mean(spec_cent)) 
    ls.append(np.var(spec_cent)) 
    ls.append(np.mean(spec_bw)) 
    ls.append(np.var(spec_bw)) 
    ls.append(np.mean(rolloff)) 
    ls.append(np.var(rolloff)) 
    ls.append(np.mean(zcr)) 
    ls.append(np.var(zcr)) 
    ls.append(np.mean(y_harm)) 
    ls.append(np.var(y_harm)) 
    ls.append(np.mean(y_perc)) 
    ls.append(np.var(y_perc)) 
    ls.append(tempo)
    for e in mfcc:
        ls.append(np.mean(e))
        ls.append(np.var(e))
    
    return ls

X = get_feature_vector('carad.wav')
get_scores(X,scaler,model)
