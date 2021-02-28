# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 19:44:48 2021

@author: dell
"""

import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.layers import Dropout
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
from keras.preprocessing.image import ImageDataGenerator
import random

from keras.preprocessing.image import load_img,img_to_array


os.makedirs('spectrograms3sec')
os.makedirs('spectrograms3sec/train')
os.makedirs('spectrograms3sec/test')

#making directories for all the genres we are considering
genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
genres = genres.split()
for g in genres:
  #path_audio = os.path.join('audio3sec',f'{g}')
  #os.makedirs(path_audio)
  path_train = os.path.join('spectrograms3sec/train',f'{g}')
  path_test = os.path.join('spectrograms3sec/test',f'{g}')
  os. makedirs(path_train)
  os. makedirs(path_test)
  
  
  
#creating 3secs clips from each audio file and putting them in respecctive directories  
from pydub import AudioSegment
i = 0
for g in genres:
  j=0
  print(f"{g}")
  for filename in os.listdir(os.path.join('genres_original',f"{g}")):
      
    try:  
        song  =  os.path.join(f'genres_original/{g}',f'{filename}')
        j = j+1
        for w in range(0,10):
          i = i+1
          #print(i)
          t1 = 3*(w)*1000
          t2 = 3*(w+1)*1000
          newAudio = AudioSegment.from_wav(song)
          new = newAudio[t1:t2]
          new.export(f'audio3sec/{g}/{g+str(j)+str(w)}.wav', format="wav")
    except:
        continue
    
#genearting spectrograms of each audio file    
for g in genres:
  j = 0
  i = 0
  print(g)
  for filename in os.listdir(os.path.join('audio3sec',f"{g}")):
    try:
        song  =  os.path.join(f'audio3sec/{g}',f'{filename}')
        j = j+1
        print(i)
        i= i+1
        y,sr = librosa.load(song,duration=3)
        #print(sr)
        mels = librosa.feature.melspectrogram(y=y,sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
        plt.savefig(f'spectrograms3sec/train/{g}/{g+str(j)}.png')
    except:
        continue

#creating test and train data
directory = "spectrograms3sec/train/"
for g in genres:
  filenames = os.listdir(os.path.join(directory,f"{g}"))
  random.shuffle(filenames)
  test_files = filenames[0:100]

  for f in test_files:

    shutil.move(directory + f"{g}"+ "/" + f,"spectrograms3sec/test/" + f"{g}")
    
#creating validation set    
train_dir = "spectrograms3sec/train/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=64)

validation_dir = "spectrograms3sec/test/"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = vali_datagen.flow_from_directory(validation_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=64)


#defining model
def GenreModel(input_shape = (288,432,4),classes=10):
 
 
  X_input = Input(input_shape)

  X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(32,kernel_size=(3,3),strides = (1,1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(256,kernel_size=(3,3),strides=(1,1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  
  X = Flatten()(X)

  #X = Dropout(rate=0.3)(X)

  #X = Dense(256,activation='relu')(X)

  #X = Dropout(rate=0.4)(X)

  X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=9))(X)

  model = Model(inputs=X_input,outputs=X,name='GenreModel')

  return model

#evaluation techniques
import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
  
model = GenreModel(input_shape=(288,432,4),classes=9)
opt = Adam(learning_rate=0.0005)
model.compile(optimizer = opt,loss='categorical_crossentropy',metrics=['accuracy',get_f1]) 

model.fit_generator(train_generator,epochs=70,validation_data=vali_generator)
model.load_weights('genre.h5')
class_labels = ['blues',
 'classical',
 'country',
 'disco',
 'hiphop',
 'metal',
 'pop',
 'reggae',
 'rock']


#helper functions to take any audio an perform the above steps to detect the genre
def convert_mp3_to_wav(music_file):
  sound = AudioSegment.from_mp3(music_file)
  sound.export("music_file.wav",format="wav")

def extract_relevant(wav_file,t1,t2):
  wav = AudioSegment.from_wav(wav_file)
  wav = wav[1000*t1:1000*t2]
  wav.export("extracted.wav",format='wav')

def create_melspectrogram(wav_file):
  y,sr = librosa.load(wav_file,duration=3)
  mels = librosa.feature.melspectrogram(y=y,sr=sr)
  
  fig = plt.Figure()
  canvas = FigureCanvas(fig)
  p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
  plt.savefig('melspectrogram.png')


def predict(image_data,model):

  #image = image_data.resize((288,432))
  image = img_to_array(image_data)

  image = np.reshape(image,(1,288,432,4))

  prediction = model.predict(image/255)

  prediction = prediction.reshape((9,)) 


  class_label = np.argmax(prediction)

  
  return class_label,prediction

def show_output(wav_file,model):
  class_labels = ['blues',
     'classical',
     'country',
     'disco',
     'hiphop',
     'metal',
     'pop',
     'reggae',
     'rock']
  create_melspectrogram(wav_file) 
  image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
  

  class_label,prediction = predict(image_data,model)
  

  prediction = prediction.reshape((9,)) 
  i=0
  for item in class_labels:
      print("score for "+ str(item) + " is "+ str(prediction[i]))
      i=i+1
  
#show the scores the audio file
sd = show_output('carad.wav',model)
