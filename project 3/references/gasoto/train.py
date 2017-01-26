from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from vgg19 import *
from TrainingData import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import model_from_json


center = 0
left = 1
right = 2
steering = 3
anglebias = 0.25
## Load data from CSV file
csvfile = open('driving_log_udacity.csv', 'rt')
lines = csvfile.readlines()    
    
X = []
y = []
X_out = []
y_out = []
for i in range(len(lines)):        
    row = lines[i].split(', ')
    cimg = row[center]       
    limg = row[left]           
    rimg = row[right]        
        
    steer = float(row[steering])
    if abs(steer) < 1e-2:
        continue
    X.append(cimg)        
    y.append(steer)
    X.append(limg)
    y.append(steer - anglebias)
    X.append(rimg)
    y.append(steer + anglebias)

X_train, y_train = shuffle(X, y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
                                                      y_train,
                                                      test_size=0.20,
                                                      random_state=42)
csvfile.close()


model = vgg((40, 160, 3));
model.compile('adam', 'mse')
model.load_weights('model - new - base 1.h5')
history = model.fit_generator(getData(X_train,y_train,128), nb_epoch=3,verbose=1,samples_per_epoch=len(X_train),validation_data = getData(X_valid,y_valid,128),nb_val_samples = len(X_valid))
with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('model.h5')

    
