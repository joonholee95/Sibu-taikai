import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling
from keras.utils import np_utils
from keras.datasets import mnist


from skimage import io
import numpy as np
from keras.utils import np_utils
from keras.models import *
from keras.layers import *

from keras import backend as K
from keras import regularizers
from keras.engine import InputSpec
from keras.constraints import Constraint
from keras.layers.convolutional import _Conv
from keras import initializers

import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.backend import tensorflow_backend as KTF


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


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)


def BuildNetwork(input_shape, class_num):
    input_img = Input(shape=input_shape)

    x = Dense(512, activation='relu')(input_img)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    Deep = Dropout(0.2)(x)
    cooc_feature0 = Cooc2D(1, (5, 5), sum_constant=1.0, max_constant=1.0)(Deep)
    cooc_feature1 = Cooc2D(20, (5, 5), sum_constant=1.0, max_constant=0.5)(Deep)
    cooc_feature2 = Cooc2D(43, (5, 5), sum_constant=1.0, max_constant=1.0 / 3.0)(Deep)
    x1 = Concatenate()([cooc_feature0, cooc_feature1, cooc_feature2])
    hlac = GlobalAveragePooling2D()(x1)
    x = Concatenate()([Deep, hlac])
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=predictions)
    return model

model = BuildNetwork((28,28,1),10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.backend import tensorflow_backend as KTF

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

fpath = './weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
tb_cb = TensorBoard(log_dir="./practice4", histogram_freq=1)
cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1, callbacks=[cp_cb,es_cb,tb_cb], validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)


KTF.set_session(old_session)

model.summary()

print(model.metrics_names)
print(score)
