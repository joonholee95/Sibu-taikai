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

# 5(lc), 6(gc) select
# batch size 16 32 48 64
#model_optimizer 1 2 3 4 -> CBF / 5 6 7 8 -> Circle

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
adam1 = Adam(lr=0.005)
adamax1 = Adamax(lr=0.005)
adagrad1 = Adagrad(lr=0.005)
adadelta1 = Adadelta(lr=0.005)
rmsprop1 = RMSprop(lr=0.005)

es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')

#1,2, -CBS
#4,5, -Circle
##################################################################################################################################################
#sgd
model_sgd_1 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_sgd_1.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_sgd_1.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_sgd_1 = model_sgd_1.evaluate(x_test, y_test, batch_size=16)

model_sgd_1.save_weights("model_sgd_1.h5")

###################################################################################################################################################
#nadam
model_nadam_1 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_nadam_1.compile(loss='categorical_crossentropy', optimizer=nadam1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_nadam_1.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_nadam_1= model_nadam_1.evaluate(x_test, y_test, batch_size=16)

model_nadam_1.save_weights("model_nadam_1.h5")

###################################################################################################################################################
#Adam
model_adam_1 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adam_1.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adam_1.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adam_1= model_adam_1.evaluate(x_test, y_test, batch_size=16)

model_adam_1.save_weights("model_adam_1.h5")

###################################################################################################################################################
#Adamax
model_adamax_1 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adamax_1.compile(loss='categorical_crossentropy', optimizer=adamax1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adamax_1.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adamax_1= model_adamax_1.evaluate(x_test, y_test, batch_size=16)

model_adamax_1.save_weights("model_adamax_1.h5")

###################################################################################################################################################
#Adagrad
model_adagrad_1 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adagrad_1.compile(loss='categorical_crossentropy', optimizer=adagrad1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adagrad_1.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adagrad_1= model_adagrad_1.evaluate(x_test, y_test, batch_size=16)

model_adagrad_1.save_weights("model_adagrad_1.h5")

###################################################################################################################################################
#Adadelta
model_adadelta_1 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adadelta_1.compile(loss='categorical_crossentropy', optimizer=adadelta1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adadelta_1.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adadelta_1= model_adadelta_1.evaluate(x_test, y_test, batch_size=16)

model_adadelta_1.save_weights("model_adadelta_1.h5")

###################################################################################################################################################
#RMSprop
model_rmsprop_1 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_rmsprop_1.compile(loss='categorical_crossentropy', optimizer=rmsprop1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_rmsprop_1.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rmsprop_1= model_rmsprop_1.evaluate(x_test, y_test, batch_size=16)

model_rmsprop_1.save_weights("model_rmsprop_1.h5")

###################################################################################################################################################

##################################################################################################################################################
#sgd
model_sgd_2 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_sgd_2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_sgd_2.fit(x_train, y_train, batch_size=32, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_sgd_2 = model_sgd_2.evaluate(x_test, y_test, batch_size=32)

model_sgd_2.save_weights("model_sgd_2.h5")

###################################################################################################################################################
#nadam
model_nadam_2 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_nadam_2.compile(loss='categorical_crossentropy', optimizer=nadam1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_nadam_2.fit(x_train, y_train, batch_size=32, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_nadam_2= model_nadam_2.evaluate(x_test, y_test, batch_size=32)

model_nadam_2.save_weights("model_nadam_2.h5")

###################################################################################################################################################
#Adam
model_adam_2 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adam_2.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adam_2.fit(x_train, y_train, batch_size=32, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adam_2= model_adam_2.evaluate(x_test, y_test, batch_size=32)

model_adam_2.save_weights("model_adam_2.h5")

###################################################################################################################################################
#Adamax
model_adamax_2 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adamax_2.compile(loss='categorical_crossentropy', optimizer=adamax1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adamax_2.fit(x_train, y_train, batch_size=32, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adamax_2= model_adamax_2.evaluate(x_test, y_test, batch_size=32)

model_adamax_2.save_weights("model_adamax_2.h5")

###################################################################################################################################################
#Adagrad
model_adagrad_2 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adagrad_2.compile(loss='categorical_crossentropy', optimizer=adagrad1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adagrad_2.fit(x_train, y_train, batch_size=32, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adagrad_2= model_adagrad_2.evaluate(x_test, y_test, batch_size=32)

model_adagrad_2.save_weights("model_adagrad_2.h5")

###################################################################################################################################################
#Adadelta
model_adadelta_2 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adadelta_2.compile(loss='categorical_crossentropy', optimizer=adadelta1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adadelta_2.fit(x_train, y_train, batch_size=32, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adadelta_2= model_adadelta_2.evaluate(x_test, y_test, batch_size=32)

model_adadelta_2.save_weights("model_adadelta_2.h5")

###################################################################################################################################################
#RMSprop
model_rmsprop_2 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_rmsprop_2.compile(loss='categorical_crossentropy', optimizer=rmsprop1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_rmsprop_2.fit(x_train, y_train, batch_size=32, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rmsprop_2= model_rmsprop_2.evaluate(x_test, y_test, batch_size=32)

model_rmsprop_2.save_weights("model_rmsprop_2.h5")


##################################################################################################################################################
#sgd
model_sgd_3 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_sgd_3.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_sgd_3.fit(x_train, y_train, batch_size=48, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_sgd_3 = model_sgd_3.evaluate(x_test, y_test, batch_size=48)

model_sgd_3.save_weights("model_sgd_3.h5")

###################################################################################################################################################
#nadam
model_nadam_3 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_nadam_3.compile(loss='categorical_crossentropy', optimizer=nadam1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_nadam_3.fit(x_train, y_train, batch_size=48, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_nadam_3= model_nadam_3.evaluate(x_test, y_test, batch_size=48)

model_nadam_3.save_weights("model_nadam_3.h5")

###################################################################################################################################################
#Adam
model_adam_3 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adam_3.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adam_3.fit(x_train, y_train, batch_size=48, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adam_3= model_adam_3.evaluate(x_test, y_test, batch_size=48)

model_adam_3.save_weights("model_adam_3.h5")

###################################################################################################################################################
#Adamax
model_adamax_3 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adamax_3.compile(loss='categorical_crossentropy', optimizer=adamax1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adamax_3.fit(x_train, y_train, batch_size=48, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adamax_3= model_adamax_3.evaluate(x_test, y_test, batch_size=48)

model_adamax_3.save_weights("model_adamax_3.h5")

###################################################################################################################################################
#Adagrad
model_adagrad_3 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adagrad_3.compile(loss='categorical_crossentropy', optimizer=adagrad1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adagrad_3.fit(x_train, y_train, batch_size=48, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adagrad_3= model_adagrad_3.evaluate(x_test, y_test, batch_size=48)

model_adagrad_3.save_weights("model_adagrad_3.h5")

###################################################################################################################################################
#Adadelta
model_adadelta_3 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adadelta_3.compile(loss='categorical_crossentropy', optimizer=adadelta1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adadelta_3.fit(x_train, y_train, batch_size=48, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adadelta_3= model_adadelta_3.evaluate(x_test, y_test, batch_size=48)

model_adadelta_3.save_weights("model_adadelta_3.h5")

###################################################################################################################################################
#RMSprop
model_rmsprop_3 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_rmsprop_3.compile(loss='categorical_crossentropy', optimizer=rmsprop1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_rmsprop_3.fit(x_train, y_train, batch_size=48, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rmsprop_3= model_rmsprop_3.evaluate(x_test, y_test, batch_size=48)

model_rmsprop_3.save_weights("model_rmsprop_3.h5")

##################################################################################################################################################
#sgd
model_sgd_4 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_sgd_4.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_sgd_4.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_sgd_4 = model_sgd_4.evaluate(x_test, y_test, batch_size=64)

model_sgd_4.save_weights("model_sgd_4.h5")

###################################################################################################################################################
#nadam
model_nadam_4 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_nadam_4.compile(loss='categorical_crossentropy', optimizer=nadam1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_nadam_4.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_nadam_4= model_nadam_4.evaluate(x_test, y_test, batch_size=64)

model_nadam_4.save_weights("model_nadam_4.h5")

###################################################################################################################################################
#Adam
model_adam_4 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adam_4.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adam_4.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adam_4= model_adam_4.evaluate(x_test, y_test, batch_size=64)

model_adam_4.save_weights("model_adam_4.h5")

###################################################################################################################################################
#Adamax
model_adamax_4 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adamax_4.compile(loss='categorical_crossentropy', optimizer=adamax1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adamax_4.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adamax_4= model_adamax_4.evaluate(x_test, y_test, batch_size=64)

model_adamax_4.save_weights("model_adamax_4.h5")

###################################################################################################################################################
#Adagrad
model_adagrad_4 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adagrad_4.compile(loss='categorical_crossentropy', optimizer=adagrad1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adagrad_4.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adagrad_4= model_adagrad_4.evaluate(x_test, y_test, batch_size=64)

model_adagrad_4.save_weights("model_adagrad_4.h5")

###################################################################################################################################################
#Adadelta
model_adadelta_4 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_adadelta_4.compile(loss='categorical_crossentropy', optimizer=adadelta1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_adadelta_4.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_adadelta_4= model_adadelta_4.evaluate(x_test, y_test, batch_size=64)

model_adadelta_4.save_weights("model_adadelta_4.h5")

###################################################################################################################################################
#RMSprop
model_rmsprop_4 = BuildNetwork6(input_shape=input_shape, class_num=3)
model_rmsprop_4.compile(loss='categorical_crossentropy', optimizer=rmsprop1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
model_rmsprop_4.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rmsprop_4= model_rmsprop_4.evaluate(x_test, y_test, batch_size=64)

model_rmsprop_4.save_weights("model_rmsprop_4.h5")


KTF.set_session(old_session)

print("model5")

print("#################16###################")
print(model_sgd_1.metrics_names)
print(score_sgd_1)
print(model_nadam_1.metrics_names)
print(score_nadam_1)
print(model_adam_1.metrics_names)
print(score_adam_1)
print(model_adamax_1.metrics_names)
print(score_adamax_1)
print(model_adagrad_1.metrics_names)
print(score_adagrad_1)
print(model_adadelta_1.metrics_names)
print(score_adadelta_1)
print(model_rmsprop_1.metrics_names)
print(score_rmsprop_1)

print("#################32###################")
print(model_sgd_2.metrics_names)
print(score_sgd_2)
print(model_nadam_2.metrics_names)
print(score_nadam_2)
print(model_adam_2.metrics_names)
print(score_adam_2)
print(model_adamax_2.metrics_names)
print(score_adamax_2)
print(model_adagrad_2.metrics_names)
print(score_adagrad_2)
print(model_adadelta_2.metrics_names)
print(score_adadelta_2)
print(model_rmsprop_2.metrics_names)
print(score_rmsprop_2)

print("#################48##################")
print(model_sgd_3.metrics_names)
print(score_sgd_3)
print(model_nadam_3.metrics_names)
print(score_nadam_3)
print(model_adam_3.metrics_names)
print(score_adam_3)
print(model_adamax_3.metrics_names)
print(score_adamax_3)
print(model_adagrad_3.metrics_names)
print(score_adagrad_3)
print(model_adadelta_3.metrics_names)
print(score_adadelta_3)
print(model_rmsprop_3.metrics_names)
print(score_rmsprop_3)


print("#################64##################")
print(model_sgd_4.metrics_names)
print(score_sgd_4)
print(model_nadam_4.metrics_names)
print(score_nadam_4)
print(model_adam_4.metrics_names)
print(score_adam_4)
print(model_adamax_4.metrics_names)
print(score_adamax_4)
print(model_adagrad_4.metrics_names)
print(score_adagrad_4)
print(model_adadelta_4.metrics_names)
print(score_adadelta_4)
print(model_rmsprop_4.metrics_names)
print(score_rmsprop_4)