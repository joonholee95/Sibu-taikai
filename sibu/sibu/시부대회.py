from MakeSynthSeries import SineWave, MakeSynthSeries
from CooccurrenceLayer import Cooc1D,SumNorm
from SimpleCooc1dNet import *
from Utilities import *
from test import test
from train import train
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Conv1D, MaxPooling1D, GlobalAveragePooling1D, LSTM, Lambda
from keras.models import Model
import numpy as np
from keras.optimizers import SGD
from keras import metrics
import argparse
import random
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.backend import tensorflow_backend as KTF

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

'''
parser = argparse.ArgumentParser()
parser.add_argument('--class_num', '-c', type=int, default=2)
args = parser.parse_args()

# Load testing data
x_test = LoadSeriesData(root_dir="./testingData/", extension=".dat")
y_test = LoadLabels(class_num=args.class_num, filename="testLabels.txt")

# Load Network
input_shape = (x_test.shape[1], 1)
print(input_shape)
print(args.class_num)
'''

#1.

test = np.loadtxt('CBF_TEST.txt',dtype=np.float,delimiter=',')
x_test = test[:,1:]+4.0
x_test = x_test[:,:,np.newaxis]
# print(x_test)
# print(x_test.shape)

train = np.loadtxt('CBF_TRAIN.txt',dtype=np.float,delimiter=',')
x_train = train[:,1:]+4.0
x_train = x_train[:,:,np.newaxis]
# print(x_train)
# print(x_train.shape)

y_test = test[:,0]
y_test = np_utils.to_categorical(y_test-1, 3)
y_train = train[:,0]
y_train = np_utils.to_categorical(y_train-1, 3)
# print(y_test)
# print(y_test.shape)
# print(y_train)

input_shape = (x_test.shape[1], 1)
# print(input_shape)

#fpath = './weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
#tb_cb = TensorBoard(log_dir="./sibu1", histogram_freq=1)
#cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')

#0.25
model = BuildNetwork14(input_shape=input_shape, class_num=3)
#sgd = SGD(lr=0.025, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=1000, verbose=1, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score = model.evaluate(x_test, y_test, batch_size=32)

KTF.set_session(old_session)

# Test
print(model.metrics_names)
print(score)

