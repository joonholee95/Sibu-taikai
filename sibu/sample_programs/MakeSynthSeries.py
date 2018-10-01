import random
from datetime import datetime
random.seed(datetime.now())
import numpy as np
import matplotlib.pyplot as plt
import os

def SineWave(length=10, #[sec]
             freq=1000, #[Hz]
             cycle=1, #[sec]
             phase=0 #[rad]
             ):
    t = np.arange(0, length, 1.0/freq, dtype=np.float)
    wave = np.sin(2*np.pi*t/cycle + phase)
    return wave

def MakeSynthSeries(list_cycle=[0.1, 0.07],
                    length=1,
                    freq = 200,
                    n_train = 10000,
                    n_test = 1000,
                    train_dir = "trainingData",
                    test_dir = "testingData",
                    extension = ".dat"):

    n_class = len(list_cycle)

    # Training data
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    for c in range(n_class):
        for n in range(n_train):
            phase = 2*np.pi*random.random()
            wave = SineWave(length=length, freq=freq, cycle=list_cycle[c], phase=phase)+1.0
            filename = train_dir + "/" + str(c*n_train + n + 1) + extension
            np.savetxt(filename, wave, fmt='%f', delimiter='\n')

    labels = np.arange(n_class).repeat(n_train)
    np.savetxt("trainLabels.txt", labels, fmt='%d', delimiter='\n')

    # Training data
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for c in range(n_class):
        for n in range(n_test):
            phase = 2*np.pi*random.random()
            wave = SineWave(length=length, freq=freq, cycle=list_cycle[c], phase=phase)+1.0
            filename = test_dir + "/" + str(c*n_test + n + 1) + extension
            np.savetxt(filename, wave, fmt='%f', delimiter='\n')

    labels = np.arange(n_class).repeat(n_test)
    np.savetxt("testLabels.txt", labels, fmt='%d', delimiter='\n')

if __name__ == '__main__':
    MakeSynthSeries()
