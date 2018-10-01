from keras.layers import Input, Dense, Flatten, Dropout, AveragePooling1D
from CooccurrenceLayer import Cooc1D
from keras.models import Model

def BuildNetwork(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Cooc1D(2, 100, activation='relu')(input_series)
    x = AveragePooling1D(pool_size=101)(x)
    x = Flatten()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model