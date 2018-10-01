import numpy as np
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

# test = np.loadtxt('CBF_TEST.txt',dtype=np.float,delimiter=',')
test = np.loadtxt('Earthquakes_TEST.txt',dtype=np.float,delimiter=',')
# test = np.loadtxt('Cricket_Y_TEST.txt',dtype=np.float,delimiter=',')
print(test.shape)
min = test.min(axis=None, keepdims=True)
max = test.max(axis=None, keepdims=True)

# train = np.loadtxt('CBF_TRAIN.txt',dtype=np.float,delimiter=',')
train = np.loadtxt('Earthquakes_TRAIN.txt',dtype=np.float,delimiter=',')
# train = np.loadtxt('Cricket_Y_TRAIN.txt',dtype=np.float,delimiter=',')
print(train.shape)

min1 = train.min(axis=None, keepdims=True)
max1 = train.max(axis=None, keepdims=True)

print(min)
print(max)

print(min1)
print(max1)

def min_max(x, axis=None):
    min = train.min(axis=axis, keepdims=True)
    max = train.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

x_train = train[:,1:]
x_train = min_max(x_train)
x_train = x_train[:,:,np.newaxis]


x_test = test[:,1:]
x_test = (min_max(x_test))
x_test = x_test[:,:,np.newaxis]



y_test = test[:,0]
# y_test = np_utils.to_categorical(y_test-1, 3)
y_test = np_utils.to_categorical(y_test-1, 2)
y_train = train[:,0]
# y_train = np_utils.to_categorical(y_train-1, 3)
y_train = np_utils.to_categorical(y_train-1, 2)

input_shape = (x_test.shape[1], 1)

sgd1 = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')


#gru + Conv
# model_gc2 = BuildNetwork61(input_shape=input_shape, class_num=12)#
# model_gc2.save_weights("model_gc(30).h5")
#

#
model_gc2 = BuildNetwork9(input_shape=input_shape, class_num=2)
# model_gc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
# model_gc2.summary()
# model_gc2.fit(x_train, y_train, batch_size=16, epochs=2000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
# score_gc2 = model_gc2.evaluate(x_test, y_test, batch_size=16)
# model_gc2.save_weights("model_gc2(0.25).h5")
model_gc2.load_weights("model_gc2(10).h5")
#
model_gc = BuildNetwork6(input_shape=input_shape, class_num=2)
# model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
# model_gc.summary()
# model_gc.fit(x_train, y_train, batch_size=16, epochs=2000, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
# score_gc = model_gc.evaluate(x_test, y_test, batch_size=16)
#
# model_gc.save_weights("model_gc(0.5).h5")
model_gc.load_weights("model_gc(10).h5")

# print(model_gc2.metrics_names)
# print(score_gc2)
# print(model_gc.metrics_names)
# print(score_gc)


def VisualizeFeature1D1(model, target_layer, input_data, target_image=0, target_image2=1 , weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    inter_model = Model(inputs=model.input, outputs=model.layers[target_layer].output)
    inter_output = inter_model.predict(input_data)
    n_data, length, n_fil = inter_output.shape
    print(inter_output.shape)
    plt.figure()
    for i in range(n_fil):
        im = inter_output[target_image, :, i]
        im2 = inter_output[target_image2, :, i]

        #im = im * max_pix_value
        plt.subplot(np.ceil((n_fil) / fig_col_num), fig_col_num, i + 1)
        plt.axis('on')
        plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        plt.tick_params(labelleft="off", left="off")  # y軸の削除
        plt.plot(im)
        plt.plot(im2)
        # plt.xlabel("Time")
        # plt.ylabel("Input value")
        # plt.rcParams["font.size"] = 14
        # plt.tight_layout()
    # plt.legend(loc='uppper right', fontsize=7)

def VisualizeFeature1D2(model, target_layer, input_data, target_image=0, target_image2=1 , weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    inter_model = Model(inputs=model.input, outputs=model.layers[target_layer].output)
    inter_output = inter_model.predict(input_data)
    n_data, length, n_fil = inter_output.shape
    print(inter_output.shape)
    plt.figure()
    for i in range(n_fil-27):
        im = inter_output[target_image, :, i]
        im2 = inter_output[target_image2, :, i]

        #im = im * max_pix_value
        plt.subplot(np.ceil((n_fil-27) / fig_col_num), fig_col_num, i + 1)
        plt.axis('on')
        # plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        # plt.tick_params(labelleft="off", left="off")  # y軸の削除
        plt.plot(im)
        plt.plot(im2)
        plt.xlabel("Time")
        plt.ylabel("Output value")
        plt.rcParams["font.size"] = 7
        plt.tight_layout()
    # plt.legend(loc='uppper right', fontsize=7)

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
        plt.ylabel("Output value")
        plt.xlabel("Time")
        # plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        # plt.tick_params(labelleft="off", left="off")  # y軸の削除
        plt.plot(im)

def VisualizeWeights1D(model, target_layer, weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    W = model.layers[target_layer].get_weights()[0]
    W = W.transpose(2, 0, 1)
    n_fil, n_dim, length = W.shape
    plt.figure()
    for i in range(n_fil-27):
        im = W[i]
        #im = im / im.max() * max_pix_value
        plt.subplot(np.ceil((n_fil-27)/ fig_col_num), fig_col_num, i + 1)
        plt.axis('on')
        plt.ylabel("Weight value")
        plt.xlabel("Filter size")
        plt.rcParams["font.size"] = 7
        plt.tight_layout()
        #plt.imshow(im, cmap='gray')
        # plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        # plt.tick_params(labelleft="off", left="off")  # y軸の削除
        plt.plot(im)


# VisualizeWeights1D(model_gc2,1, weight_name='(e)model_gc2(16).h5',fig_col_num=3)
# VisualizeFeature1D(model_hg,1,x_test,target_image=128, weight_name='model_gc(10).h5',fig_col_num=1) #0
# VisualizeFeature1D(model_gc,1,x_test,target_image=22, weight_name='(e)model_gc(16).h5',fig_col_num=1) #7
# VisualizeFeature1D(model_gc,1,x_test,target_image=40, weight_name='(e)model_gc(16).h5',fig_col_num=1) #7
# VisualizeFeature1D(model_gc,1,x_test,target_image=54, weight_name='(e)model_gc(16).h5',fig_col_num=1) #7

# plt.figure()
# plt.rcParams["font.size"] = 14
# plt.tight_layout()
# p1=plt.plot(x_test[0],label='Class 0')
# p2=plt.plot(x_test[1],label='Class 1')
# plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
# plt.tick_params(labelleft="off", left="off")  # y軸の削除
# p2=plt.plot(x_test[46], label= 'Class 11')
# plt.xlabel("Time")
# plt.ylabel("Input value")
# plt.legend(fontsize=14, loc = 1)
# plt.rcParams["font.size"] = 14
# plt.tight_layout()

# VisualizeFeature1D1(model_gc,1,x_test,target_image=0 , target_image2=1, weight_name='model_gc(10).h5',fig_col_num=5)
# VisualizeFeature1D2(model_gc,1,x_test,target_image=0 , target_image2=1, weight_name='model_gc(10).h5',fig_col_num=1)
# VisualizeFeature1D1(model_gc2,1,x_test,target_image=0 , target_image2=1, weight_name='model_gc2(10).h5',fig_col_num=5)
# VisualizeFeature1D2(model_gc2,1,x_test,target_image=0 , target_image2=1, weight_name='model_gc2(10).h5',fig_col_num=1)
# VisualizeFeature1D(model_hg,1,x_test,target_image=187, weight_name='(e)model_gc(16).h5',fig_col_num=3) #10
# VisualizeFeature1D(model_hg,1,x_test,target_image=174, weight_name='(e)model_gc(16).h5',fig_col_num=3) #11
# VisualizeFeature1D(model_gc,1,x_test,target_image=120, weight_name='model_gc(10).h5',fig_col_num=1) #10
VisualizeFeature1D(model_gc2,1,x_test,target_image=120, weight_name='model_gc2(10).h5',fig_col_num=1) #10
VisualizeWeights1D(model_gc,1, weight_name='model_gc(10).h5',fig_col_num=1)
VisualizeWeights1D(model_gc,1, weight_name='model_gc(10).h5',fig_col_num=1)
VisualizeWeights1D(model_gc2,1, weight_name='model_gc2(10).h5',fig_col_num=1)

plt.show()

