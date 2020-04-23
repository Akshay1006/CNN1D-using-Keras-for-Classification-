#Use the train_norm from data preparation to run the CNN 1D Across Rows

#Model Building-CNN1D 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D, AveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from keras.models import load_model
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import os

no_of_vars=1 # No of independent features in the models
event_len=train_norm.shape[1] # To transpose the data
unique_accts=train_df.shape[0] # Number of Unique Accounts

train_norm_reshape=train_norm.reshape(unique_accts,event_len,no_of_vars)

conv1=Sequential()
conv1.add(Conv1D(50,5,input_shape=(train_norm_reshape.shape[1],train_norm_reshape.shape[2]),activation='relu'))
conv1.add(MaxPooling1D(3))
conv1.add(Conv1D(30,5,activation='relu',padding='same'))
conv1.add(MaxPooling1D(3))
conv1.add(Flatten())
conv1.add(Dense(512,activation='relu'))
conv1.add(Dropout(0.3))
conv1.add(Dense(256,activation='relu'))
conv1.add(Dropout(0.3))
conv1.add(Dense(64,activation='relu'))
conv1.add(Dropout(0.2))
conv1.add(Dense(32,activation='relu'))
conv1.add(Dropout(0.2))
conv1.add(Dense(train_dep_data.shape[1],kernel_initializer='uniform',activation='sigmoid'))

conv1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

conv1.summary()

conv1.fit(train_norm_reshape, train_dep_data, epochs =10 ,steps_per_epoch=15, verbose = True)

conv1.save('./cnn1d_across_rows.h5')

#Predict the Output for the Out of Time data

no_of_vars=1 # No of independent features in the models
event_len=oot_norm.shape[1] # To transpose the data
unique_accts=oot_df.shape[0] # Number of Unique Accounts

oot_norm_reshape=oot_norm.reshape(unique_accts,event_len,no_of_vars)

pred_oot=model.predict_proba(oot_norm_reshape)
