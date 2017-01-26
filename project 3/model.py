from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import cv2, numpy as np

def nvidia():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(64,64,3)))

    model.add(Convolution2D(24, 5, 5), border_mode='valid', activation='relu', subsample=(2,2))
    model.add(Convolution2D(36, 5, 5), border_mode='valid', activation='relu', subsample=(2,2))
    model.add(Convolution2D(48, 3, 3), border_mode='valid', activation='relu', subsample=(2,2))
    model.add(Convolution2D(64, 3, 3), border_mode='valid', activation='relu', subsample=(2,2))
    model.add(Convolution2D(64, 3, 3), border_mode='valid', activation='relu', subsample=(2,2))

    model.add(Flatten())
    model.add(Dense(1164), activation='relu')
    model.add(Dense(100), activation='relu')
    model.add(Dense(50), activation='relu')
    model.add(Dense(10), activation='relu')
    model.add(Dense(1), activation='relu')
