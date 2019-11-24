# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:41:43 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 00:17:44 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:40:25 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:11:22 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 08:54:10 2019

@author: User
"""

import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as sk_mae
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dont allocate entire vram initially
set_session(tf.Session(config=config))
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Sequential
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
IMG_SIZE = (224, 224) 
def flow_from_dataframe(in_df, path_col, y_col, **dflow_args):
    img_data_gen = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = False, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'reflect',
                              zoom_range=0.25,
                             preprocessing_function = preprocess_input)
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode = 'sparse',**dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
print("Reading data...")
img_dir = "dataset/"
csv_path = "Dataset.csv"
df = pd.read_csv(csv_path)
df['path'] = df['Id'].map(lambda x: img_dir+"{}.jpg".format(x))
df['exists'] = df['path'].map(os.path.exists)
print("{} images found out of total {} images".format(df['exists'].sum(),df.shape[0]))
print("Reading complete !!!\n")
print("Preparing training, testing and validation datasets ...")
raw_train_df, test_df = train_test_split(df, 
                                   test_size = 0.30, 
                                   random_state = 2019,
                                   stratify = df['Sign'])
raw_train_df, valid_df = train_test_split(raw_train_df, 
                                   test_size = 0.30,
                                   random_state = 2019,
                                   stratify = raw_train_df['Sign'])
train_df=raw_train_df
train_size=train_df.shape[0]
valid_size=valid_df.shape[0]
test_size=test_df.shape[0]
print("# Training images:   {}".format(train_size))
print("# Validation images: {}".format(valid_size))
print("# Testing images:    {}".format(test_size))
      
train_gen = flow_from_dataframe(train_df, 
                             path_col = 'path',
                            y_col = 'Sign', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 10)
valid_X, valid_Y = next(flow_from_dataframe(valid_df, 
                             path_col = 'path',
                            y_col = 'Sign', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = valid_size))


IMG_SHAPE = valid_X[0,:,:,:].shape
print("Image shape: "+str(IMG_SHAPE))

base_iv3_model = InceptionV3(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')
model = Sequential()
model.add(base_iv3_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu')) 

model.compile(optimizer = 'adam', loss = 'mse')
model.summary()
weight_path="inceptionv3.best.hdf5" 
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list = [checkpoint, early, reduceLROnPlat]

if not os.path.exists(weight_path):
    model.fit_generator(train_gen, 
                                  steps_per_epoch=500,
                                  validation_data = (valid_X, valid_Y), 
                                  epochs = 10, 
                                  callbacks = callbacks_list,
                                  verbose=1)
else:
    model.load_weights(weight_path)
test_X, test_Y = next(flow_from_dataframe(test_df, 
                             path_col = 'path',
                            y_col = 'Sign', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = test_size))

pred_Y = model.predict(test_X,batch_size=25,verbose=1)
print("Mean absolute error on test data: "+str(sk_mae(test_Y,pred_Y)))