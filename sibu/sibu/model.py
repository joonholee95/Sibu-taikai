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

#model-> (rnn , lstm, gru) ->( 1 , cooc , Conv ), hlac + deep Cnn, lenet + cooc, hlac + gru
#############################################################################################################################################

#rnn
model_rnn = BuildNetwork1(input_shape=input_shape, class_num=12)
model_rnn.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rnn.summary()
model_rnn.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rnn = model_rnn.evaluate(x_test, y_test, batch_size=32)

model_rnn.save_weights("model_rnn(20).h5")

##############################################################################################################################################
#lstm
model_lstm = BuildNetwork2(input_shape=input_shape, class_num=12)
model_lstm.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lstm.summary()
model_lstm.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lstm = model_lstm.evaluate(x_test, y_test, batch_size=32)

model_lstm.save_weights("model_lstm(20).h5")

##############################################################################################################################################
#gru
model_gru = BuildNetwork3(input_shape=input_shape, class_num=12)
model_gru.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gru.summary()
model_gru.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gru = model_gru.evaluate(x_test, y_test, batch_size=32)

model_gru.save_weights("model_gru(20).h5")

##############################################################################################################################################

#rnn + cooc
model_rc = BuildNetwork4(input_shape=input_shape, class_num=12)
model_rc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc.summary()
model_rc.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc = model_rc.evaluate(x_test, y_test, batch_size=32)

model_rc.save_weights("model_rc(20).h5")

##############################################################################################################################################
#lstm + cooc
model_lc = BuildNetwork5(input_shape=input_shape, class_num=12)
model_lc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc.summary()
model_lc.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc = model_lc.evaluate(x_test, y_test, batch_size=32)

model_lc.save_weights("model_lc(20).h5")

##############################################################################################################################################

#gru + cooc
model_gc = BuildNetwork6(input_shape=input_shape, class_num=12)
model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc.summary()
model_gc.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc = model_gc.evaluate(x_test, y_test, batch_size=32)

model_gc.save_weights("model_gc(20).h5")

##############################################################################################################################################
#rnn + Conv
model_rc2 = BuildNetwork7(input_shape=input_shape, class_num=12)
model_rc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc2.summary()
model_rc2.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc2 = model_rc2.evaluate(x_test, y_test, batch_size=32)

model_rc2.save_weights("model_rc2(20).h5")

##############################################################################################################################################
#lstm + Conv
model_lc2 = BuildNetwork8(input_shape=input_shape, class_num=12)
model_lc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc2.summary()
model_lc2.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc2 = model_lc2.evaluate(x_test, y_test, batch_size=32)

model_lc2.save_weights("model_lc2(20).h5")

##############################################################################################################################################
#gru + Conv
model_gc2 = BuildNetwork9(input_shape=input_shape, class_num=12)
model_gc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc2.summary()
model_gc2.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc2 = model_gc2.evaluate(x_test, y_test, batch_size=32)

model_gc2.save_weights("model_gc2(20).h5")

#############################################################################################################################################
#hlac + deep cnn
model_hd = BuildNetwork10(input_shape=input_shape, class_num=12)
model_hd.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hd.summary()
model_hd.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hd = model_hd.evaluate(x_test, y_test, batch_size=32)

model_hd.save_weights("model_hd(20).h5")

##############################################################################################################################################
#cooc +lenet
model_cl = BuildNetwork11(input_shape=input_shape, class_num=12)
model_cl.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_cl.summary()
model_cl.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_cl = model_cl.evaluate(x_test, y_test, batch_size=32)

model_cl.save_weights("model_c1(20).h5")

##############################################################################################################################################
#hlac + gru
model_hg = BuildNetwork12(input_shape=input_shape, class_num=12)
model_hg.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hg.summary()
model_hg.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hg = model_hg.evaluate(x_test, y_test, batch_size=32)

model_hg.save_weights("model_hg(20).h5")

##############################################################################################################################################


print("--------------32--------------")
print(model_rnn.metrics_names)
print(score_rnn)
print(model_lstm.metrics_names)
print(score_lstm)
print(model_gru.metrics_names)
print(score_gru)
print(model_rc.metrics_names)
print(score_rc)
print(model_lc.metrics_names)
print(score_lc)
print(model_gc.metrics_names)
print(score_gc)
print(model_rc2.metrics_names)
print(score_rc2)
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


#VisualizeFeature1D(model,1, input_data= x_test, target_image=49, weight_name='model14(20).h5')
#VisualizeWeights1D(model,1, weight_name='model14(20).h5')

#rnn
model_rnn = BuildNetwork1(input_shape=input_shape, class_num=12)
model_rnn.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rnn.summary()
model_rnn.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rnn = model_rnn.evaluate(x_test, y_test, batch_size=48)

model_rnn.save_weights("model_rnn(21).h5")

##############################################################################################################################################
#lstm
model_lstm = BuildNetwork2(input_shape=input_shape, class_num=12)
model_lstm.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lstm.summary()
model_lstm.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lstm = model_lstm.evaluate(x_test, y_test, batch_size=48)

model_lstm.save_weights("model_lstm(21).h5")

##############################################################################################################################################
#gru
model_gru = BuildNetwork3(input_shape=input_shape, class_num=12)
model_gru.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gru.summary()
model_gru.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gru = model_gru.evaluate(x_test, y_test, batch_size=48)

model_gru.save_weights("model_gru(21).h5")

##############################################################################################################################################

#rnn + cooc
model_rc = BuildNetwork4(input_shape=input_shape, class_num=12)
model_rc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc.summary()
model_rc.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc = model_rc.evaluate(x_test, y_test, batch_size=48)

model_rc.save_weights("model_rc(21).h5")

##############################################################################################################################################
#lstm + cooc
model_lc = BuildNetwork5(input_shape=input_shape, class_num=12)
model_lc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc.summary()
model_lc.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc = model_lc.evaluate(x_test, y_test, batch_size=48)

model_lc.save_weights("model_lc(21).h5")

##############################################################################################################################################

#gru + cooc
model_gc = BuildNetwork6(input_shape=input_shape, class_num=12)
model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc.summary()
model_gc.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc = model_gc.evaluate(x_test, y_test, batch_size=48)

model_gc.save_weights("model_gc(21).h5")

##############################################################################################################################################
#rnn + Conv
model_rc2 = BuildNetwork7(input_shape=input_shape, class_num=12)
model_rc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc2.summary()
model_rc2.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc2 = model_rc2.evaluate(x_test, y_test, batch_size=48)

model_rc2.save_weights("model_rc2(21).h5")

##############################################################################################################################################
#lstm + Conv
model_lc2 = BuildNetwork8(input_shape=input_shape, class_num=12)
model_lc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc2.summary()
model_lc2.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc2 = model_lc2.evaluate(x_test, y_test, batch_size=48)

model_lc2.save_weights("model_lc2(21).h5")

##############################################################################################################################################
#gru + Conv
model_gc2 = BuildNetwork9(input_shape=input_shape, class_num=12)
model_gc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc2.summary()
model_gc2.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc2 = model_gc2.evaluate(x_test, y_test, batch_size=48)

model_gc2.save_weights("model_gc2(21).h5")

#############################################################################################################################################
#hlac + deep cnn
model_hd = BuildNetwork10(input_shape=input_shape, class_num=12)
model_hd.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hd.summary()
model_hd.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hd = model_hd.evaluate(x_test, y_test, batch_size=48)

model_hd.save_weights("model_hd(21).h5")

##############################################################################################################################################
#cooc +lenet
model_cl = BuildNetwork11(input_shape=input_shape, class_num=12)
model_cl.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_cl.summary()
model_cl.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_cl = model_cl.evaluate(x_test, y_test, batch_size=48)

model_cl.save_weights("model_c1(21).h5")

##############################################################################################################################################
#hlac + gru
model_hg = BuildNetwork12(input_shape=input_shape, class_num=12)
model_hg.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hg.summary()
model_hg.fit(x_train, y_train, batch_size=48, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hg = model_hg.evaluate(x_test, y_test, batch_size=48)

model_hg.save_weights("model_hg(21).h5")

##############################################################################################################################################

print("--------------48--------------")
print(model_rnn.metrics_names)
print(score_rnn)
print(model_lstm.metrics_names)
print(score_lstm)
print(model_gru.metrics_names)
print(score_gru)
print(model_rc.metrics_names)
print(score_rc)
print(model_lc.metrics_names)
print(score_lc)
print(model_gc.metrics_names)
print(score_gc)
print(model_rc2.metrics_names)
print(score_rc2)
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

##############################################################################################################################################

#rnn
model_rnn = BuildNetwork1(input_shape=input_shape, class_num=12)
model_rnn.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rnn.summary()
model_rnn.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rnn = model_rnn.evaluate(x_test, y_test, batch_size=64)

model_rnn.save_weights("model_rnn(22).h5")

##############################################################################################################################################
#lstm
model_lstm = BuildNetwork2(input_shape=input_shape, class_num=12)
model_lstm.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lstm.summary()
model_lstm.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lstm = model_lstm.evaluate(x_test, y_test, batch_size=64)

model_lstm.save_weights("model_lstm(22).h5")

##############################################################################################################################################
#gru
model_gru = BuildNetwork3(input_shape=input_shape, class_num=12)
model_gru.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gru.summary()
model_gru.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gru = model_gru.evaluate(x_test, y_test, batch_size=64)

model_gru.save_weights("model_gru(22).h5")

##############################################################################################################################################

#rnn + cooc
model_rc = BuildNetwork4(input_shape=input_shape, class_num=12)
model_rc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc.summary()
model_rc.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc = model_rc.evaluate(x_test, y_test, batch_size=64)

model_rc.save_weights("model_rc(22).h5")

##############################################################################################################################################
#lstm + cooc
model_lc = BuildNetwork5(input_shape=input_shape, class_num=12)
model_lc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc.summary()
model_lc.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc = model_lc.evaluate(x_test, y_test, batch_size=64)

model_lc.save_weights("model_lc(22).h5")

##############################################################################################################################################

#gru + cooc
model_gc = BuildNetwork6(input_shape=input_shape, class_num=12)
model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc.summary()
model_gc.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc = model_gc.evaluate(x_test, y_test, batch_size=64)

model_gc.save_weights("model_gc(22).h5")

##############################################################################################################################################
#rnn + Conv
model_rc2 = BuildNetwork7(input_shape=input_shape, class_num=12)
model_rc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc2.summary()
model_rc2.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc2 = model_rc2.evaluate(x_test, y_test, batch_size=64)

model_rc2.save_weights("model_rc2(22).h5")

##############################################################################################################################################
#lstm + Conv
model_lc2 = BuildNetwork8(input_shape=input_shape, class_num=12)
model_lc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc2.summary()
model_lc2.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc2 = model_lc2.evaluate(x_test, y_test, batch_size=64)

model_lc2.save_weights("model_lc2(22).h5")

##############################################################################################################################################
#gru + Conv
model_gc2 = BuildNetwork9(input_shape=input_shape, class_num=12)
model_gc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc2.summary()
model_gc2.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc2 = model_gc2.evaluate(x_test, y_test, batch_size=64)

model_gc2.save_weights("model_gc2(22).h5")

#############################################################################################################################################
#hlac + deep cnn
model_hd = BuildNetwork10(input_shape=input_shape, class_num=12)
model_hd.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hd.summary()
model_hd.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hd = model_hd.evaluate(x_test, y_test, batch_size=64)

model_hd.save_weights("model_hd(22).h5")

##############################################################################################################################################
#cooc +lenet
model_cl = BuildNetwork11(input_shape=input_shape, class_num=12)
model_cl.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_cl.summary()
model_cl.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_cl = model_cl.evaluate(x_test, y_test, batch_size=64)

model_cl.save_weights("model_c1(22).h5")

##############################################################################################################################################
#hlac + gru
model_hg = BuildNetwork12(input_shape=input_shape, class_num=12)
model_hg.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hg.summary()
model_hg.fit(x_train, y_train, batch_size=64, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hg = model_hg.evaluate(x_test, y_test, batch_size=64)

model_hg.save_weights("model_hg(22).h5")

##############################################################################################################################################

print("--------------64--------------")
print(model_rnn.metrics_names)
print(score_rnn)
print(model_lstm.metrics_names)
print(score_lstm)
print(model_gru.metrics_names)
print(score_gru)
print(model_rc.metrics_names)
print(score_rc)
print(model_lc.metrics_names)
print(score_lc)
print(model_gc.metrics_names)
print(score_gc)
print(model_rc2.metrics_names)
print(score_rc2)
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


##############################################################################################################################################

#rnn
model_rnn = BuildNetwork1(input_shape=input_shape, class_num=12)
model_rnn.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rnn.summary()
model_rnn.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rnn = model_rnn.evaluate(x_test, y_test, batch_size=16)

model_rnn.save_weights("model_rnn(19).h5")

##############################################################################################################################################
#lstm
model_lstm = BuildNetwork2(input_shape=input_shape, class_num=12)
model_lstm.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lstm.summary()
model_lstm.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lstm = model_lstm.evaluate(x_test, y_test, batch_size=16)

model_lstm.save_weights("model_lstm(19).h5")

##############################################################################################################################################
#gru
model_gru = BuildNetwork3(input_shape=input_shape, class_num=12)
model_gru.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gru.summary()
model_gru.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gru = model_gru.evaluate(x_test, y_test, batch_size=16)

model_gru.save_weights("model_gru(19).h5")

##############################################################################################################################################

#rnn + cooc
model_rc = BuildNetwork4(input_shape=input_shape, class_num=12)
model_rc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc.summary()
model_rc.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc = model_rc.evaluate(x_test, y_test, batch_size=16)

model_rc.save_weights("model_rc(19).h5")

##############################################################################################################################################
#lstm + cooc
model_lc = BuildNetwork5(input_shape=input_shape, class_num=12)
model_lc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc.summary()
model_lc.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc = model_lc.evaluate(x_test, y_test, batch_size=16)

model_lc.save_weights("model_lc(19).h5")

##############################################################################################################################################

#gru + cooc
model_gc = BuildNetwork6(input_shape=input_shape, class_num=12)
model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc.summary()
model_gc.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc = model_gc.evaluate(x_test, y_test, batch_size=16)

model_gc.save_weights("model_gc(19).h5")

##############################################################################################################################################
#rnn + Conv
model_rc2 = BuildNetwork7(input_shape=input_shape, class_num=12)
model_rc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc2.summary()
model_rc2.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc2 = model_rc2.evaluate(x_test, y_test, batch_size=16)

model_rc2.save_weights("model_rc2(19).h5")

##############################################################################################################################################
#lstm + Conv
model_lc2 = BuildNetwork8(input_shape=input_shape, class_num=12)
model_lc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc2.summary()
model_lc2.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc2 = model_lc2.evaluate(x_test, y_test, batch_size=16)

model_lc2.save_weights("model_lc2(19).h5")

##############################################################################################################################################
#gru + Conv
model_gc2 = BuildNetwork9(input_shape=input_shape, class_num=12)
model_gc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc2.summary()
model_gc2.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc2 = model_gc2.evaluate(x_test, y_test, batch_size=16)

model_gc2.save_weights("model_gc2(19).h5")

#############################################################################################################################################
#hlac + deep cnn
model_hd = BuildNetwork10(input_shape=input_shape, class_num=12)
model_hd.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hd.summary()
model_hd.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hd = model_hd.evaluate(x_test, y_test, batch_size=16)

model_hd.save_weights("model_hd(19).h5")

##############################################################################################################################################
#cooc +lenet
model_cl = BuildNetwork11(input_shape=input_shape, class_num=12)
model_cl.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_cl.summary()
model_cl.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_cl = model_cl.evaluate(x_test, y_test, batch_size=16)

model_cl.save_weights("model_c1(19).h5")

##############################################################################################################################################
#hlac + gru
model_hg = BuildNetwork12(input_shape=input_shape, class_num=12)
model_hg.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hg.summary()
model_hg.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hg = model_hg.evaluate(x_test, y_test, batch_size=16)

model_hg.save_weights("model_hg(19).h5")

##############################################################################################################################################
KTF.set_session(old_session)

print("--------------16--------------")
print(model_rnn.metrics_names)
print(score_rnn)
print(model_lstm.metrics_names)
print(score_lstm)
print(model_gru.metrics_names)
print(score_gru)
print(model_rc.metrics_names)
print(score_rc)
print(model_lc.metrics_names)
print(score_lc)
print(model_gc.metrics_names)
print(score_gc)
print(model_rc2.metrics_names)
print(score_rc2)
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
