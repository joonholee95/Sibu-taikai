from keras.layers import Input, Dense, Flatten, Dropout, AveragePooling1D, GlobalAveragePooling1D, LSTM,Concatenate,GRU,average, Lambda, BatchNormalization,MaxPooling1D,Conv1D, Bidirectional, RNN ,SimpleRNN
from CooccurrenceLayer import Cooc1D
from keras.models import Model
from keras.backend import mean,max

from Utilities import *
from modeling import *
from test import test
import numpy as np
from keras.optimizers import SGD , nadam, Adam, Adamax, Adagrad, Adadelta, RMSprop
from keras import metrics
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

#test = np.loadtxt('Cricket_Y_TEST.txt',dtype=np.float,delimiter=',')
test = np.loadtxt('CBF_TEST.txt',dtype=np.float,delimiter=',')

def min_max(x, axis=None):
    min = test.min(axis=axis, keepdims=True)
    max = test.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

x_test = test[:,1:]
x_test = (min_max(x_test))
x_test = x_test[:,:,np.newaxis]
#train = np.loadtxt('Cricket_Y_TRAIN.txt',dtype=np.float,delimiter=',')
train = np.loadtxt('CBF_TRAIN.txt',dtype=np.float,delimiter=',')

x_train = train[:,1:]
x_train = min_max(x_train)
x_train = x_train[:,:,np.newaxis]

y_test = test[:,0]
y_test = np_utils.to_categorical(y_test-1, 3)
#y_test = np_utils.to_categorical(y_test-1, 12)
y_train = train[:,0]
y_train = np_utils.to_categorical(y_train-1, 3)
#y_train = np_utils.to_categorical(y_train-1, 12)

input_shape = (x_test.shape[1], 1)

#fpath = './weight4/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
#cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

sgd1 = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
nadam1 = nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')


#hd
def BuildNetwork10(input_shape, class_num):
    input_series = Input(shape=input_shape)
    cooc_feature0 = Cooc1D(1, 96, sum_constant=1.0, max_constant=1.0)(input_series)
    cooc_feature1 = Cooc1D(20, 96, sum_constant=1.0, max_constant=0.5)(input_series)
    cooc_feature2 = Cooc1D(43, 96, sum_constant=1.0, max_constant=1.0/3.0)(input_series)
    hlac = Concatenate()([cooc_feature0, cooc_feature1, cooc_feature2])
    hlac = GlobalAveragePooling1D()(hlac)
    x = Conv1D(64, 64, activation='relu')(input_series)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = GlobalAveragePooling1D()(x)
    x = Concatenate()([x, hlac])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#c1
def BuildNetwork11(input_shape, class_num):
    input_series = Input(shape=input_shape)
    cooc_feature0 = Cooc1D(1, 96, sum_constant=1.0, max_constant=1.0)(input_series)
    cooc_feature1 = Cooc1D(20, 96, sum_constant=1.0, max_constant=0.5)(input_series)
    cooc_feature2 = Cooc1D(43, 96, sum_constant=1.0, max_constant=1.0 / 3.0)(input_series)
    x = Concatenate()([cooc_feature0, cooc_feature1, cooc_feature2])
    hlac = GlobalAveragePooling1D()(x)
    x = Conv1D(20, 96, activation='relu', padding='same')(input_series)
    x = MaxPooling1D(2)(x)
    x = Conv1D(50, 96, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    x = Concatenate()([x, hlac])
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#hg
def BuildNetwork12(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x1 = Cooc1D(1, 96, activation='relu',sum_constant=1.0, max_constant=1.0)(input_series)
    x2 = Cooc1D(20, 96, activation='relu',sum_constant=1.0, max_constant=0.5)(input_series)
    x3 = Cooc1D(43, 96, activation='relu',sum_constant=1.0, max_constant=1.0/3.0)(input_series)
    hlac= Concatenate()([x1, x2, x3])
    hlac = GlobalAveragePooling1D()(hlac)
    x = GRU(128)(input_series)
    x = Dense(64,activation='relu')(x)
    x = Concatenate()([x, hlac])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model

#############################################################################################################


model = BuildNetwork10(input_shape=input_shape, class_num=3)
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
nadam1 = nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score = model.evaluate(x_test, y_test, batch_size=32)

model = BuildNetwork11(input_shape=input_shape, class_num=3)
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
nadam1 = nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score = model.evaluate(x_test, y_test, batch_size=32)

model = BuildNetwork12(input_shape=input_shape, class_num=3)
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
nadam1 = nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score = model.evaluate(x_test, y_test, batch_size=32)

