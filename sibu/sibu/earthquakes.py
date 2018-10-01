from Utilities import *
#from model import *
from keras.layers import *
from test import test
import numpy as np
from keras.optimizers import SGD , nadam, Adam, Adamax, Adagrad, Adadelta, RMSprop
from keras import metrics
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf
from modeling import *
from CooccurrenceLayer import Cooc1D
#1,2 -> CBF
#3,4 -> Circle
#5,6 -> Earthquakes 64.5%


old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

# test = np.loadtxt('Earthquakes_TEST.txt',dtype=np.float,delimiter=',')
test = np.loadtxt('Cricket_Y_TEST.txt',dtype=np.float,delimiter=',')
#test = np.loadtxt('CBF_TEST.txt',dtype=np.float,delimiter=',')
def min_max(x, axis=None):
    min = train.min(axis=axis, keepdims=True)
    max = train.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

# train = np.loadtxt('Earthquakes_TRAIN.txt',dtype=np.float,delimiter=',')
train = np.loadtxt('Cricket_Y_TRAIN.txt',dtype=np.float,delimiter=',')
#train = np.loadtxt('CBF_TRAIN.txt',dtype=np.float,delimiter=',')

x_train = train[:,1:]
x_train = min_max(x_train)
x_train = x_train[:,:,np.newaxis]


x_test = test[:,1:]
x_test = (min_max(x_test))
x_test = x_test[:,:,np.newaxis]



y_test = test[:,0]
#y_test = np_utils.to_categorical(y_test-1, 3)
y_test = np_utils.to_categorical(y_test-1, 12)

y_train = train[:,0]
#y_train = np_utils.to_categorical(y_train-1, 3)
y_train = np_utils.to_categorical(y_train-1, 12)

input_shape = (x_test.shape[1], 1)

sgd1 = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')


#fpath = './weight4/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
#cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#1; gru + conv -. 98.3
#2: simple rnn ->98.3
#3:
#4:
#batch -> 32 64 16(6)
#model-> (rnn , lstm, gru) ->( 1 , cooc , Conv ), hlac + deep Cnn, lenet + cooc, hlac + gru




##############################################################################################################################################
#gru + cooc
model_gc = BuildNetwork9(input_shape=input_shape, class_num=12)
model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc.summary()
model_gc.fit(x_train, y_train, batch_size=32, epochs=200, verbose=1, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc = model_gc.evaluate(x_test, y_test, batch_size=32)



##############################################################################################################################################

print(model_gc.metrics_names)
print(score_gc)

##############################################################################################################################################
