import argparse
import gc
import random
from datetime import datetime
random.seed(datetime.now())
from keras.optimizers import SGD
from keras import metrics
from SimpleCooc1dNet import BuildNetwork
from Utilities import LoadSeriesData, LoadLabels

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num', '-c', type=int, default=2)
    parser.add_argument('--n_epoch', '-e', type=int, default=50)
    args = parser.parse_args()

    # Load training data
    x_train = LoadSeriesData(root_dir="./trainingData/", extension=".dat")
    y_train = LoadLabels(class_num=args.class_num, filename="trainLabels.txt")

    # Build network
    input_shape = (x_train.shape[1], 1)
    model = BuildNetwork(input_shape=input_shape, class_num=args.class_num)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy])
    model.summary()

    # Training
    model.fit(x_train, y_train, batch_size=32, epochs=args.n_epoch, shuffle=True)
    print("finished")
    del x_train, y_train
    gc.collect()
    model.save_weights("weights")

if __name__ == '__main__':
    train()