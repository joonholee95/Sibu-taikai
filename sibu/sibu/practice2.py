
from keras.models import *
from keras.layers import *

from skimage import io
import numpy as np
from keras.utils import np_utils, plot_model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf
from keras import regularizers
from keras.engine import InputSpec
from keras.constraints import Constraint
from keras.layers.convolutional import _Conv
from keras import initializers
import os
old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

TrainingSampleNum = 2000  # 学習サンプル総数
TestSampleNum = 10000  # テストサンプル総数
ClassNum = 10  # クラス数（今回は10）
ImageSize = 28  # 画像サイズ（今回は縦横ともに28）
TrainingDataFile = '/home/jinho/Desktop/Python/Images/TrainingSamples/{0:1d}-{1:04d}.png'
TestDataFile = '/home/jinho/Desktop/Python/Images/TestSamples/{0:1d}-{1:04d}.png'
OutFile = '/home/jinho/Desktop/Python/Images/OutSamples/gray_{0:1d}-{1:04d}.png'

# ImageProcessing ルーチン
def ImageProcessing(src):
    '''
    ここでは画素の操作方法がわかるようにあえて２重ループで書いている．
    実際には以下の１行で同じ機能が実現できる．

    dest = src // 2
    '''

    dest = np.zeros(src.shape, dtype=np.uint8)
    for y in range(0, src.shape[0]):
        for x in range(0, src.shape[1]):
            dest[y, x] = src[y, x] // 2

    return dest


# Binarization ルーチン
def Binarization(src, thres):
    '''
    ここでは画素の操作方法がわかるようにあえて２重ループで書いている．
    '''

    dest = np.zeros(src.shape, dtype=np.uint8)
    for y in range(0, src.shape[0]):
        for x in range(0, src.shape[1]):
            if src[y, x] < thres:
                dest[y, x] = 255
            else:
                dest[y, x] = 0

    return dest



class SumNorm(Constraint):
    def __init__(self, sum_const=1.0, max_const=0.5, axis=0):
        self.sum_const = sum_const
        self.max_const = max_const
        self.axis = axis

    def __call__(self, w):
        w = K.clip(w, K.epsilon(), self.max_const)
        sum_val = K.sum(w, axis=self.axis, keepdims=True) + K.epsilon() #To prevent division by zero
        w = w / sum_val * self.sum_const
        return w

    def get_config(self):
        return {'sum_const': self.sum_const,
                'max_const': self.max_const,
                'axis': self.axis}


class Cooc2D(_Conv):
    def __init__(self, filters,
                 kernel_size,
                 sum_constant=1.0,
                 max_constant=0.5,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer=initializers.random_uniform(minval=0.0,maxval=1.0),
                 bias_initializer='zeros',
                 kernel_regularizer=regularizers.l1(0.01),
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(Cooc2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=SumNorm(sum_const=sum_constant, max_const=max_constant, axis=[0,1,2]),
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs):
        log_input = K.log(inputs + K.epsilon())
        innner_prod = K.conv2d(
            log_input,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        outputs = K.exp(innner_prod)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(Cooc2D, self).get_config()
        config.pop('rank')
        return config



# main ルーチン

y_train = []
x_train = []

for label in range(0, ClassNum):
    for sample in range(0, TrainingSampleNum // ClassNum):
        filename = TrainingDataFile.format(label, sample)
        print("Loading the file: " + filename)
        img = io.imread(filename)
        img = Binarization(img, 200)
        x_train.append(img)
        y_train.append(label)

x_train = np.array(x_train)
x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train, 10)


y_test = []
x_test = []

for label in range(0, ClassNum):
    for sample in range(0, TestSampleNum // ClassNum):
        filename = TestDataFile.format(label, sample)
        print("Loading the file: " + filename)
        img = io.imread(filename)
        img = Binarization(img, 200)
        x_test.append(img)
        y_test.append(label)

x_test = np.array(x_test)
x_test = x_test.astype('float32')
x_test /= 255

y_test = np_utils.to_categorical(y_test, 10)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

def BuildNetwork(input_shape, class_num):
    input_img = Input(shape=input_shape)
    cooc_feature0 = Cooc2D(1, (5, 5), sum_constant=1.0, max_constant=1.0)(input_img)
    cooc_feature1 = Cooc2D(20, (5, 5), sum_constant=1.0, max_constant=0.5)(input_img)
    cooc_feature2 = Cooc2D(43, (5, 5), sum_constant=1.0, max_constant=1.0 / 3.0)(input_img)
    x = Concatenate()([cooc_feature0, cooc_feature1, cooc_feature2])
    hlac = GlobalAveragePooling2D()(x)
    x = Conv2D(20, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(50, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Concatenate()([x, hlac])
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=predictions)
    return model

model = BuildNetwork((28,28,1),10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


fpath = './weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
tb_cb = TensorBoard(log_dir="./practice2", histogram_freq=1)
cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


model.fit(x_train, y_train, batch_size=32, nb_epoch=10, verbose=1, callbacks=[cp_cb,es_cb,tb_cb], validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

KTF.set_session(old_session)

model.summary()

print(model.metrics_names)
print(score)
