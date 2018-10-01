from keras.layers import Input, Dense, Flatten, Dropout, AveragePooling1D, GlobalAveragePooling1D, LSTM,Concatenate,GRU,average, Lambda, BatchNormalization,MaxPooling1D,Conv1D, Bidirectional, RNN ,SimpleRNN
from CooccurrenceLayer import Cooc1D
from keras.models import Model
from keras.backend import mean,max

#model-> (rnn , lstm, gru) ->( 1 , cooc , Conv ), hlac + deep Cnn, lenet + cooc, hlac + gru

#############################################################################################################
#rnn
def BuildNetwork1(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = SimpleRNN(64)(input_series)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model

#############################################################################################################
#lstm
def BuildNetwork2(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = LSTM(64)(input_series)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#gru
def BuildNetwork3(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = GRU(64)(input_series)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#rc
def BuildNetwork4(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Cooc1D(30, 96, activation='relu', sum_constant=1.0, max_constant=1.0)(input_series)
    x = AveragePooling1D(2)(x)
    x = SimpleRNN(64)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#lc
def BuildNetwork5(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Cooc1D(30, 96, activation='relu', sum_constant=1.0, max_constant=1.0)(input_series)
    x = AveragePooling1D(2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#gc
def BuildNetwork6(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Cooc1D(30, 96, activation='relu', sum_constant=1.0, max_constant=0.2)(input_series)
    x = AveragePooling1D(2)(x)
    x = GRU(64)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model

def BuildNetwork61(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Cooc1D(30, 96, activation='relu', sum_constant=1.0, max_constant=0.5)(input_series)
    x = AveragePooling1D(2)(x)
    x = GRU(64)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model


#############################################################################################################
#rc2
def BuildNetwork7(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Conv1D(30, 96, activation='relu')(input_series)
    x = AveragePooling1D(2)(x)
    x = SimpleRNN(64)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#lc2
def BuildNetwork8(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Conv1D(30, 96, activation='relu')(input_series)
    x = AveragePooling1D(2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#gc2
def BuildNetwork9(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Conv1D(30, 96, activation='relu',bias_initializer="RandomNormal")(input_series)
    x = AveragePooling1D(2)(x)
    x = GRU(64)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model

def BuildNetwork91(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x = Conv1D(30, 96, activation='relu')(input_series)
    # x = AveragePooling1D(2)(x)
    x = GRU(64)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#hd
def BuildNetwork10(input_shape, class_num):
    input_series = Input(shape=input_shape)
    cooc_feature0 = Cooc1D(1, 96, sum_constant=1.0, max_constant=1.0)(input_series)
    cooc_feature1 = Cooc1D(20, 96, sum_constant=1.0, max_constant=0.5)(input_series)
    cooc_feature2 = Cooc1D(43, 96, sum_constant=1.0, max_constant=1.0/3.0)(input_series)
    hlac = Concatenate()([cooc_feature0, cooc_feature1, cooc_feature2])
    hlac = GlobalAveragePooling1D()(hlac)
    x = Conv1D(64, 64, activation='relu')(input_series)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = GlobalAveragePooling1D()(x)
    x = Concatenate()([x, hlac])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#c1
def BuildNetwork11(input_shape, class_num):
    input_series = Input(shape=input_shape)
    cooc_feature0 = Cooc1D(1, 96, sum_constant=1.0, max_constant=1.0)(input_series)
    cooc_feature1 = Cooc1D(20, 96, sum_constant=1.0, max_constant=0.5)(input_series)
    cooc_feature2 = Cooc1D(43, 96, sum_constant=1.0, max_constant=1.0 / 3.0)(input_series)
    x = Concatenate()([cooc_feature0, cooc_feature1, cooc_feature2])
    hlac = GlobalAveragePooling1D()(x)
    x = Conv1D(20, 96, activation='relu', padding='same')(input_series)
    x = MaxPooling1D(2)(x)
    x = Conv1D(50, 96, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    x = Concatenate()([x, hlac])
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################
#hg
def BuildNetwork12(input_shape, class_num):
    input_series = Input(shape=input_shape)
    x1 = Cooc1D(1, 96, activation='relu',sum_constant=1.0, max_constant=1.0)(input_series)
    x2 = Cooc1D(20, 96, activation='relu',sum_constant=1.0, max_constant=0.5)(input_series)
    x3 = Cooc1D(43, 96, activation='relu',sum_constant=1.0, max_constant=1.0/3.0)(input_series)
    hlac= Concatenate()([x1, x2, x3])
    hlac = GlobalAveragePooling1D()(hlac)
    x = GRU(128)(input_series)
    x = Dense(64,activation='relu')(x)
    x = Concatenate()([x, hlac])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input_series, outputs=predictions)
    return model
#############################################################################################################



