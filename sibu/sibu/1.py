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

#1,2,3 4-> CBF
#5,6,7,8 -> Circle
#9,10,11,12 -> earth
#batch size(CBF) -> 32, 16 ,48, 64
#model_name 2 3 4 / 5 6

old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

#test = np.loadtxt('Earthquakes_TEST.txt',dtype=np.float,delimiter=',')
test = np.loadtxt('Cricket_Y_TEST.txt',dtype=np.float,delimiter=',')
#test = np.loadtxt('CBF_TEST.txt',dtype=np.float,delimiter=',')
def min_max(x, axis=None):
    min = test.min(axis=axis, keepdims=True)
    max = test.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

#train = np.loadtxt('Earthquakes_TRAIN.txt',dtype=np.float,delimiter=',')
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

sgd1 = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')


##############################################################################################################################################
#lstm + cooc
model_lc = BuildNetwork5(input_shape=input_shape, class_num=2)
model_lc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc.summary()
model_lc.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc = model_lc.evaluate(x_test, y_test, batch_size=64)

model_json_lc = model_lc.to_json()
with open("(e)model_lc", "w") as json_file :
    json_file.write(model_json_lc)

model_lc.save_weights("(e)model_lc(64).h5")

##############################################################################################################################################
#gru + cooc
model_gc = BuildNetwork6(input_shape=input_shape, class_num=2)
model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc.summary()
model_gc.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc = model_gc.evaluate(x_test, y_test, batch_size=64)

model_json_gc = model_gc.to_json()
with open("(e)model_gc", "w") as json_file :
    json_file.write(model_json_gc)

model_gc.save_weights("(e)model_gc(64).h5")


##############################################################################################################################################
#lstm + Conv
model_lc2 = BuildNetwork8(input_shape=input_shape, class_num=2)
model_lc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc2.summary()
model_lc2.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc2 = model_lc2.evaluate(x_test, y_test, batch_size=64)

model_json_lc2 = model_lc2.to_json()
with open("(e)model_lc2", "w") as json_file :
    json_file.write(model_json_lc2)

model_lc2.save_weights("(e)model_lc2(64).h5")

##############################################################################################################################################
#gru + Conv
model_gc2 = BuildNetwork9(input_shape=input_shape, class_num=2)
model_gc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc2.summary()
model_gc2.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc2 = model_gc2.evaluate(x_test, y_test, batch_size=64)

model_json_gc2 = model_gc2.to_json()
with open("(e)model_gc2", "w") as json_file :
    json_file.write(model_json_gc2)

model_gc2.save_weights("(e)model_gc2(64).h5")

#############################################################################################################################################
#hlac + deep cnn
model_hd = BuildNetwork10(input_shape=input_shape, class_num=2)
model_hd.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hd.summary()
model_hd.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hd = model_hd.evaluate(x_test, y_test, batch_size=64)

model_json_hd = model_hd.to_json()
with open("(e)model_hd", "w") as json_file :
    json_file.write(model_json_hd)

model_hd.save_weights("(e)model_hd(64).h5")

##############################################################################################################################################
#cooc +lenet
model_cl = BuildNetwork11(input_shape=input_shape, class_num=2)
model_cl.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_cl.summary()
model_cl.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_cl = model_cl.evaluate(x_test, y_test, batch_size=64)

model_json_cl = model_cl.to_json()
with open("(e)model_cl", "w") as json_file :
    json_file.write(model_json_cl)

model_cl.save_weights("(e)model_c1(64).h5")

##############################################################################################################################################
#hlac + gru
model_hg = BuildNetwork12(input_shape=input_shape, class_num=2)
model_hg.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hg.summary()
model_hg.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hg = model_hg.evaluate(x_test, y_test, batch_size=64)

model_json_hg = model_hg.to_json()
with open("(e)model_hg", "w") as json_file :
    json_file.write(model_json_hg)

model_hg.save_weights("(e)model_hg(64).h5")

##############################################################################################################################################

print(model_lc.metrics_names)
print(score_lc)
print(model_gc.metrics_names)
print(score_gc)
print(model_lc2.metrics_names)
print(score_lc2)
print(model_gc2.metrics_names)
print(score_gc2)
print(model_hd.metrics_names)
print(score_hd)
print(model_cl.metrics_names)
print(score_cl)
print(model_hg.metrics_names)
print(score_hg)
