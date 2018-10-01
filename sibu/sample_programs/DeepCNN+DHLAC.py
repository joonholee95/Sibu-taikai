from keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from CooccurrenceLayer import Cooc2D
from keras.models import Model

def BuildNetwork(input_shape, class_num):
    input_img = Input(shape=input_shape)
    cooc_feature0 = Cooc2D(1, (5, 5), sum_constant=1.0, max_constant=1.0)(input_img)
    cooc_feature1 = Cooc2D(20, (5, 5), sum_constant=1.0, max_constant=0.5)(input_img)
    cooc_feature2 = Cooc2D(43, (5, 5), sum_constant=1.0, max_constant=1.0/3.0)(input_img)
    x = Concatenate()([cooc_feature0, cooc_feature1, cooc_feature2])
    hlac = GlobalAveragePooling2D()(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Concatenate()([x, hlac])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=predictions)
    return model