# 모델 구조의 비교 -> lstm rnn gru CuDNNLSTM CuDNNGRU (Conv + Cooc)
# 최적화 함수의 비교 -> SGD, Adam, Nadam, Adamax, Adagrad, Adadelta, RMSprop, TFOptimizer

from Utilities import *
from SimpleCooc1dNet import *
from test import test
import numpy as np
from keras.optimizers import SGD , nadam, Adam, Adamax, Adagrad, Adadelta, RMSprop, TFOptimizer
from keras import metrics
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

#test = np.loadtxt('CBF_TEST.txt',dtype=np.float,delimiter=',')
test = np.loadtxt('Cricket_Y_TEST.txt',dtype=np.float,delimiter=',')

def min_max(x, axis=None):
    min = test.min(axis=axis, keepdims=True)
    max = test.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

x_test = test[:,1:]
x_test = (min_max(x_test))
x_test = x_test[:,:,np.newaxis]
#print(x_test.shape)

#train = np.loadtxt('CBF_TRAIN.txt',dtype=np.float,delimiter=',')
train = np.loadtxt('Cricket_Y_TRAIN.txt',dtype=np.float,delimiter=',')

x_train = train[:,1:]
x_train = min_max(x_train)
x_train = x_train[:,:,np.newaxis]
#print(x_train.shape)

y_test = test[:,0]
#y_test = np_utils.to_categorical(y_test-1, 3)
y_test = np_utils.to_categorical(y_test-1, 12)

y_train = train[:,0]
#y_train = np_utils.to_categorical(y_train-1, 3)
y_train = np_utils.to_categorical(y_train-1, 12)
#print(y_test.shape)

input_shape = (x_test.shape[1], 1)
print(input_shape)
#path setting

#weight, sibu -> cooc1d sgd 97.1
#weight1, sibu1 -> conv1d sgd 98.1
#weight2, sibu2 -> conv1d nadam other data 0.08
#weight3, sibu3 -> cooc1d nadam other data 0.07
#weight4, sibu4 -> conv1d sgd other data 0.07
#weight5, sibu5 -> cooc1d sgd other data 0.60
fpath = './weight4/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
#fpath1 = './weight2/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'

tb_cb = TensorBoard(log_dir="./sibu4", histogram_freq=1)
#tb_cb1 = TensorBoard(log_dir="./sibu2", histogram_freq=1)

cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#cp_cb1 = ModelCheckpoint(filepath=fpath1, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')

#정규화를 하고 나서부터는 학습률이 떨어져 sgd로 학습률를 올리고 배치사이지를 크게하고 에폭 수를 많이해 학습을 많이해  97.3정도는 나오게 됬지만 정규화를 하기전이 더 정도가 높다 (배치사이즈 64 애폭 500 학습률 0.005 early 100)> 왜?
#0.25
model = BuildNetwork14(input_shape=input_shape, class_num=12)
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
nadam1 = nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model.summary()

model.fit(x_train, y_train, batch_size=8, epochs=1000, verbose=1, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score = model.evaluate(x_test, y_test, batch_size=8)
'''''
model_json1 = model.to_json()
with open("model14", "w") as json_file :
    json_file.write(model_json1)

model.save_weights("model14.h5")
print("Saved model to disk")
'''''
KTF.set_session(old_session)

#VisualizeFeature1D(model,1, input_data= x_test, target_image=49, weight_name='model14.h5')
#VisualizeWeights1D(model,1, weight_name='model14.h5')


print(model.metrics_names)
print(score)

#8,9 -> 97.888
