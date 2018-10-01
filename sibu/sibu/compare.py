
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from keras.models import model_from_json ,model_from_yaml
from test import test
import numpy as np
from CooccurrenceLayer import Cooc1D
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf
from keras import metrics
from keras.utils import np_utils
from keras.optimizers import SGD
from modeling import *

#old_session = KTF.get_session()
#session = tf.Session('')
#KTF.set_session(session)
#KTF.set_learning_phase(1)

# test = np.loadtxt('Earthquakes_TEST.txt',dtype=np.float,delimiter=',')
test = np.loadtxt('Cricket_Y_TEST.txt',dtype=np.float,delimiter=',')
#test = np.loadtxt('CBF_TEST.txt',dtype=np.float,delimiter=',')
def min_max(x, axis=None):
    min = test.min(axis=axis, keepdims=True)
    max = test.max(axis=axis, keepdims=True)
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

print(x_test.shape)

y_test = test[:,0]
#y_test = np_utils.to_categorical(y_test-1, 3)
y_test = np_utils.to_categorical(y_test-1, 12)

y_train = train[:,0]
#y_train = np_utils.to_categorical(y_train-1, 3)
y_train = np_utils.to_categorical(y_train-1, 12)

input_shape = (x_test.shape[1], 1)

sgd1 = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
# es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

model_gc = BuildNetwork6(input_shape=input_shape, class_num=12)
model_gc.load_weights("(e)model_gc(16).h5")
# model_gc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
# score_gc2 = model_gc2.evaluate(x_test, y_test, batch_size=16)
# model_hg.summary()

model_gc2 = BuildNetwork9(input_shape=input_shape, class_num=12)
model_gc2.load_weights("(e)model_gc2(16).h5")
# model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
# score_gc = model_gc.evaluate(x_test, y_test, batch_size=16)

# print(score_gc)
# print(score_gc2)

def VisualizeFeature1D1(model, target_layer, input_data, target_image=0, target_image2=1 , weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    inter_model = Model(inputs=model.input, outputs=model.layers[target_layer].output)
    inter_output = inter_model.predict(input_data)
    n_data, length, n_fil = inter_output.shape
    # print(inter_output.shape)
    plt.figure()
    for i in range(n_fil):
        im = inter_output[target_image, :, i]
        im2 = inter_output[target_image2, :, i]

        # im = im * max_pix_value
        plt.subplot(np.ceil((n_fil) / fig_col_num), fig_col_num, i + 1)
        plt.axis('on')
        plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        plt.tick_params(labelleft="off", left="off")  # y軸の削除
        plt.plot(im)
        plt.plot(im2)
        # plt.xlabel("Time")
        # plt.ylabel("Output value")
        # plt.rcParams["font.size"] = 10
        # plt.tight_layout()
    # plt.legend(loc='uppper right', fontsize=4)


def VisualizeFeature1D(model, target_layer, input_data, target_image=0, weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    inter_model = Model(inputs=model.input, outputs=model.layers[target_layer].output)
    inter_output = inter_model.predict(input_data)
    n_data, length, n_fil = inter_output.shape
    print(inter_output.shape)
    plt.figure()
    for i in range(n_fil):
        im = inter_output[target_image, :, i]
        #im = im * max_pix_value
        plt.subplot(np.ceil((n_fil) / fig_col_num), fig_col_num, i + 1)
        plt.axis('on')
        # plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        # plt.tick_params(labelleft="off", left="off")  # y軸の削除
        plt.plot(im)

def VisualizeWeights1D(model, target_layer, weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    W = model.layers[target_layer].get_weights()[0]
    W = W.transpose(2, 0, 1)
    n_fil, n_dim, length = W.shape
    plt.figure()
    for i in range(n_fil):
        im = W[i]
        #im = im / im.max() * max_pix_value
        plt.subplot(np.ceil((n_fil)/ fig_col_num), fig_col_num, i + 1)
        plt.axis('on')
        #plt.imshow(im, cmap='gray')
        plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        plt.tick_params(labelleft="off", left="off")  # y軸の削除
        plt.plot(im)
        # plt.xlabel("Filter size")
        # plt.ylabel("Weight value")
        # plt.rcParams["font.size"] = 10
        # plt.tight_layout()

# VisualizeWeights1D(model_gc2,1, weight_name='(e)model_gc2(16).h5',fig_col_num=3)
# VisualizeFeature1D(model_hg,1,x_test,target_image=128, weight_name='model_gc(10).h5',fig_col_num=1) #0
# VisualizeFeature1D(model_gc,1,x_test,target_image=22, weight_name='(e)model_gc(16).h5',fig_col_num=1) #7
# VisualizeFeature1D(model_gc,1,x_test,target_image=40, weight_name='(e)model_gc(16).h5',fig_col_num=1) #7
# VisualizeFeature1D(model_gc,1,x_test,target_image=54, weight_name='(e)model_gc(16).h5',fig_col_num=1) #7

# plt.figure()
# plt.rcParams["font.size"] = 14
# plt.tight_layout()
# p1=plt.plot(x_test[3], color = 'c')
# plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
# plt.tick_params(labelleft="off", left="off")  # y軸の削除
# p2=plt.plot(x_test[46], label= 'Class 11')
# plt.xlabel("Time")
# plt.ylabel("Input value")
# plt.legend(fontsize=14, loc = 1)
# plt.rcParams["font.size"] = 14
# plt.tight_layout()

VisualizeFeature1D1(model_gc,1,x_test,target_image=31 , target_image2=46, weight_name='(e)model_gc(16).h5',fig_col_num=5)
VisualizeFeature1D1(model_gc2,1,x_test,target_image=31 , target_image2=46, weight_name='(e)model_gc2(16).h5',fig_col_num=5)
# VisualizeFeature1D(model_gc,1,x_test,target_image=31, weight_name='(e)model_gc(16).h5',fig_col_num=5) #10
# VisualizeFeature1D(model_gc,1,x_test,target_image=46, weight_name='(e)model_gc(16).h5',fig_col_num=5) #11
# VisualizeFeature1D(model_gc2,1,x_test,target_image=184, weight_name='(e)model_gc2(16).h5',fig_col_num=3) #10
VisualizeWeights1D(model_gc,1, weight_name='(e)model_gc(16).h5',fig_col_num=5)
VisualizeWeights1D(model_gc2,1, weight_name='(e)model_gc2(16).h5',fig_col_num=5)

plt.show()

#40097038660

