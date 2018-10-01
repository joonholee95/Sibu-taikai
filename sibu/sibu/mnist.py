#=====================================
# Co-occurrence layer
# Author: HAYASHI Hideaki
# Last update: 2017/11/13
#=====================================

from keras import backend as K
from keras import regularizers
from keras.engine import InputSpec
from keras.constraints import Constraint
from keras.layers.convolutional import _Conv
from keras import initializers

#=====================================
#Sum constraint
# Constraints the weights to have a sum to be equal to a constant.
# Argument
#   sum_const: Float, constant value which the norm of weights is equal to.
#   max_const: Float, each weight value is clipped to this value if weight > max_const
#   axis: integer, axis along which to calculate weight norms.
# Return
#   w: normalized weight tensor
#=====================================
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


#=====================================
#2D Co-occurrence layer
# Arguments
#        filters: Integer, the dimensionality of the output space
#            (i.e. the number output of filters in the convolution).
#        kernel_size: An integer or tuple/list of 2 integers, specifying the
#            width and height of the 2D convolution window.
#            Can be a single integer to specify the same value for
#            all spatial dimensions.
#        sum_const: Float, constant value which the norm of weights is equal to.
#        max_const: Float, each weight value is clipped to this value if weight > max_const
#        strides: An integer or tuple/list of 2 integers,
#            specifying the strides of the convolution along the width and height.
#            Can be a single integer to specify the same value for
#            all spatial dimensions.
#            Specifying any stride value != 1 is incompatible with specifying
#            any `dilation_rate` value != 1.
#        padding: one of `"valid"` or `"same"` (case-insensitive).
#        data_format: A string,
#            one of `channels_last` (default) or `channels_first`.
#            The ordering of the dimensions in the inputs.
#            `channels_last` corresponds to inputs with shape
#            `(batch, height, width, channels)` while `channels_first`
#            corresponds to inputs with shape
#            `(batch, channels, height, width)`.
#            It defaults to the `image_data_format` value found in your
#            Keras config file at `~/.keras/keras.json`.
#            If you never set it, then it will be "channels_last".
#        dilation_rate: an integer or tuple/list of 2 integers, specifying
#            the dilation rate to use for dilated convolution.
#            Can be a single integer to specify the same value for
#            all spatial dimensions.
#            Currently, specifying any `dilation_rate` value != 1 is
#            incompatible with specifying any stride value != 1.
#        activation: Activation function to use
#            (see [activations](../activations.md)).
#            If you don't specify anything, no activation is applied
#            (ie. "linear" activation: `a(x) = x`).
#        use_bias: Boolean, whether the layer uses a bias vector.
#        kernel_initializer: Initializer for the `kernel` weights matrix
#            (see [initializers](../initializers.md)).
#        bias_initializer: Initializer for the bias vector
#            (see [initializers](../initializers.md)).
#        kernel_regularizer: Regularizer function applied to
#            the `kernel` weights matrix
#            (see [regularizer](../regularizers.md)).
#        bias_regularizer: Regularizer function applied to the bias vector
#            (see [regularizer](../regularizers.md)).
#        activity_regularizer: Regularizer function applied to
#            the output of the layer (its "activation").
#            (see [regularizer](../regularizers.md)).
#        kernel_constraint: Constraint function applied to the kernel matrix
#            (see [constraints](../constraints.md)).
#        bias_constraint: Constraint function applied to the bias vector
#            (see [constraints](../constraints.md)).
# Input shape
#        4D tensor with shape:
#        `(samples, channels, rows, cols)` if data_format='channels_first'
#        or 4D tensor with shape:
#        `(samples, rows, cols, channels)` if data_format='channels_last'.
# Output shape
#        4D tensor with shape:
#        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
#        or 4D tensor with shape:
#        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
#        `rows` and `cols` values might have changed due to padding.
#=====================================
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


#=====================================
#1D Co-occurrence layer
# Arguments
#        filters: Integer, the dimensionality of the output space
#            (i.e. the number output of filters in the convolution).
#        kernel_size: An integer or tuple/list of a single integer, specifying the
#            length of the 1D convolution window.
#        sum_const: Float, constant value which the norm of weights is equal to.
#        max_const: Float, each weight value is clipped to this value if weight > max_const
#        strides: An integer or tuple/list of a single integer,
#            specifying the strides of the convolution along the length.
#            Specifying any stride value != 1 is incompatible with specifying
#            any `dilation_rate` value != 1.
#        padding: one of `"valid"` or `"same"` (case-insensitive).
#        dilation_rate: an integer or tuple/list of 2 integers, specifying
#            the dilation rate to use for dilated convolution.
#            Can be a single integer to specdify the same value for
#            all spatial dimensions.
#            Currently, specifying any `dilation_rate` value != 1 is
#            incompatible with specifying any stride value != 1.
#        activation: Activation function to use
#            (see [activations](../activations.md)).
#            If you don't specify anything, no activation is applied
#            (ie. "linear" activation: `a(x) = x`).
#        use_bias: Boolean, whether the layer uses a bias vector.
#        kernel_initializer: Initializer for the `kernel` weights matrix
#            (see [initializers](../initializers.md)).
#        bias_initializer: Initializer for the bias vector
#            (see [initializers](../initializers.md)).
#        kernel_regularizer: Regularizer function applied to
#            the `kernel` weights matrix
#            (see [regularizer](../regularizers.md)).
#        bias_regularizer: Regularizer function applied to the bias vector
#            (see [regularizer](../regularizers.md)).
#        activity_regularizer: Regularizer function applied to
#            the output of the layer (its "activation").
#            (see [regularizer](../regularizers.md)).
#        kernel_constraint: Constraint function applied to the kernel matrix
#            (see [constraints](../constraints.md)).
#        bias_constraint: Constraint function applied to the bias vector
#            (see [constraints](../constraints.md)).
# Input shape
#        3D tensor with shape: `(batch_size, steps, input_dim)`
# Output shape
#         3D tensor with shape: `(batch_size, new_steps, filters)`
#        `steps` value might have changed due to padding or strides.
#=====================================

class Cooc1D(_Conv):
    def __init__(self, filters,
                 kernel_size,
                 sum_constant=1.0,
                 max_constant=0.5,
                 strides=1,
                 padding='valid',
                 activation=None,
                 use_bias=False,
                 kernel_initializer=initializers.random_uniform(minval=0.0,maxval=1.0),
                 bias_initializer='zeros',
                 kernel_regularizer=regularizers.l1(0.01),
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(Cooc1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=SumNorm(sum_const=sum_constant, max_const=max_constant, axis=[0,1]),
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def call(self, inputs):
        log_input = K.log(inputs + K.epsilon())
        innner_prod = K.conv1d(
            log_input,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format)
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
        config = super(Cooc1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config



import numpy as np

np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling
from keras.utils import np_utils

from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Cooc2D(32,3,3,input_shape=(28,28,1)))
model.add(Conv2D(32, 3, 3, activation='relu'))
print(model.output_shape)

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names)
print(score)