import argparse
import random
from datetime import datetime
random.seed(datetime.now())
from Utilities import LoadSeriesData, LoadLabels
from keras.optimizers import SGD
from keras import metrics
from SimpleCooc1dNet import BuildNetwork

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num', '-c', type=int, default=2)
    args = parser.parse_args()

    # Load testing data
    x_test = LoadSeriesData(root_dir="./testingData/", extension=".dat")
    y_test = LoadLabels(class_num=args.class_num, filename="testLabels.txt")

    # Load Network
    input_shape = (x_test.shape[1], 1)
    model = BuildNetwork(input_shape=input_shape, class_num=args.class_num)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy])
    model.load_weights("weights")

    # Test
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)

if __name__ == '__main__':
    test()

