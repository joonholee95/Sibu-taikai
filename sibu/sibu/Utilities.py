import os
import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Model

def LoadData(root_dir="./Images/", extension=".png", color=False):
    file_list = os.listdir(root_dir)
    data_num = len(file_list)
    data = []
    for n in range(data_num):
        in_file_name = root_dir + str(n+1) + extension
        im = Image.open(in_file_name)
        im = np.asarray(im).tolist()
        data.append(im)
    data = np.array(data)
    data = data.astype('float32') / 255.
    if color==False:
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
    return data

def LoadSeriesData(root_dir="./Data/", extension=".dat"):
    file_list = os.listdir(root_dir)
    data_num = len(file_list)
    data = []
    for n in range(data_num):
        in_file_name = root_dir + str(n+1) + extension
        series = np.loadtxt(in_file_name, delimiter="\t")
        series = np.asarray(series).tolist()
        data.append(series)
    data = np.array(data)
    if np.ndim(data) == 2:
        data = data[:,:,np.newaxis]
    return data

def LoadLabels(class_num, filename="labels.txt"):
    labels = np.loadtxt(filename, delimiter="\n").astype(np.int64)
    labels = to_categorical(labels, class_num)
    return labels

def VisualizeWeights(model, target_layer, weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    W = model.layers[target_layer].get_weights()[0]
    W = W.transpose(3, 2, 0, 1)
    n_fil, n_ch, n_row, n_col = W.shape
    plt.figure()
    for i in range(n_fil):
        im = W[i, 0]
        im = im / im.max() * max_pix_value
        plt.subplot(np.ceil(n_fil / fig_col_num), fig_col_num, i + 1)
        plt.axis('off')
        plt.imshow(im, cmap='gray')
    plt.show()

def VisualizeWeights1D(model, target_layer, weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    W = model.layers[target_layer].get_weights()[0]
    W = W.transpose(2, 0, 1)
    n_fil, n_dim, length = W.shape
    plt.figure()
    for i in range(n_fil):
        im = W[i]
        #im = im / im.max() * max_pix_value
        plt.subplot(np.ceil(n_fil / fig_col_num), fig_col_num, i + 1)
        plt.axis('on')
        #plt.imshow(im, cmap='gray')
        plt.plot(im)
    # plt.show()

def VisualizeFeature(model, target_layer, input_data, target_image=0, weight_name="weights", fig_col_num=8, max_pix_value=255, color='gray'):
    model.load_weights(weight_name)
    inter_model = Model(inputs=model.input, outputs=model.layers[target_layer].output)
    inter_output = inter_model.predict(input_data)
    n_data, n_col, n_row, n_fil = inter_output.shape
    plt.figure()
    for i in range(n_fil):
        im = inter_output[target_image, :, :, i]
        im = im * max_pix_value
        plt.subplot(np.ceil(n_fil / fig_col_num), fig_col_num, i + 1)
        plt.axis('off')
        plt.imshow(im, cmap=color)
    plt.show()

def VisualizeFeature1D(model, target_layer, input_data, target_image=0, weight_name="weights", fig_col_num=8, max_pix_value=255):
    model.load_weights(weight_name)
    inter_model = Model(inputs=model.input, outputs=model.layers[target_layer].output)
    inter_output = inter_model.predict(input_data)
    n_data, length, n_fil = inter_output.shape
    plt.figure()
    for i in range(n_fil):
        im = inter_output[target_image, :, i]
        #im = im * max_pix_value
        plt.subplot(np.ceil(n_fil / fig_col_num), fig_col_num, i + 1)
        plt.axis('on')
        plt.plot(im)
    # plt.show()

def CalcAverageFeature(model, target_layer, input_data, start_sample=0, end_sample=0, weight_name="weights"):
    model.load_weights(weight_name)
    inter_model = Model(inputs=model.input, outputs=model.layers[target_layer].output)
    inter_output = inter_model.predict(input_data)
    return inter_output[start_sample:end_sample].mean(axis=0)
