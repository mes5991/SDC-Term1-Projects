import data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

if 0:
    '''Run line below to resave training and testing CSV files'''
    data.load_csv()

#Get training data
X_train, y_train = data.get_training_data()

#Display an image
if 0:
    plt.figure()
    plt.imshow(X_train[0])
    plt.show()

#Print out image shape
print('Image shape:', X_train[0].shape)

#One-hot encode the labels
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
print('Labels One-hot encoded')



X_train = np.array(X_train)
y_one_hot = np.array(y_one_hot)
print(type(X_train))
raw_input('break')

#Build model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))
print("Model Built")

#Compile model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
print('Model Compiled')

#Train model
print('Training...')
history = model.fit(X_train, y_one_hot, nb_epoch = 10, validation_split = 0.2)
print('Model Trained!')
