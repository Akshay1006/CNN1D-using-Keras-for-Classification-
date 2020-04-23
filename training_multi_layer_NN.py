#Use the train_norm from data preparation to run the simple neural network

#Model Building-Multi Layer Neural Network 

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

model = Sequential()
model.add(Dense(256, input_dim=train_norm.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(train_dep_data.shape[1],kernel_initializer='uniform',activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

model.fit(train_norm, train_dep_data, epochs =5 ,batch_size= 10000, verbose = True)

model.save('./base_nn.h5')

#Predict the Output for the Out of Time data

pred_oot=model.predict_proba(oot_norm)
