import csv
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def load_csv():
    '''Function to load csv data, split into training and testing sets, and
    and save those sets to a csv'''

    # data_path = "udacityData/data/driving_log.csv" #ubuntu path
    data_path = "data/udacityData/data/driving_log.csv" #windows path

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        csvRows = []
        leftImagePaths = []
        rightImagePaths = []
        centerImagePaths = []
        steeringValue = []
        row = next(reader)
        print(row)
        for row in reader: #format: center name, left name, right name, steering, throttle, brake, speed
            # path = 'udacityData/data/' #ubuntu path
            path = 'data/udacityData/data/' #windows path
            row[0] = path + row[0]
            row[1] = path + row[1]
            row[2] = path + row[2]
            csvRows.append(row)
            # centerImagePaths.append(path + row[0])
            # leftImagePaths.append(path + row[1])
            # rightImagePaths.append(path + row[2])
            # steeringValue.append(row[3])

    #Split into training and testing sets
    csvTrain, csvTest = train_test_split(csvRows, test_size = 0.2)
    # trainingPath = 'udacityData/data/training.csv' #ubuntu path
    # testingPath = 'udacityData/data/testing.csv' #ubuntu path
    trainingPath = 'data/udacityData/data/training.csv' #windows path
    testingPath = 'data/udacityData/data/testing.csv' #windows path
    with open(trainingPath, 'w', newline='') as trainingFile, open(testingPath, 'w', newline='') as testingFile:
        writer = csv.writer(trainingFile)
        writer.writerows(csvTrain)
        writer = csv.writer(testingFile)
        writer.writerows(csvTest)

    print('Training csv saved to:', trainingPath, 'Size:', len(csvTrain), 'Percent:', len(csvTrain)/(len(csvTrain)+len(csvTest)))
    print('Testing csv saved to:', testingPath, 'Size:', len(csvTest), 'Percent:', len(csvTest)/(len(csvTrain)+len(csvTest)))

def get_training_data():
    '''Helper function to return Udacity training data'''

    # path = 'udacityData/data/training.csv' #ubuntu path
    path = 'data/udacityData/data/training.csv' #windows path
    X_csv = []
    y_train = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X_csv.append(row[0]) #center image
            y_train.append(row[3]) #steering angle
    print('Training CSV loaded')
    X_train = [mpimg.imread(imPath) for imPath in X_csv]
    print('Training images loaded')
    return(X_train, y_train)

def get_testing_data():
    '''Helper function to return Udacity testing data'''

    # path = 'udacityData/data/testing.csv' #ubuntu path
    path = 'data/udacityData/data/testing.csv' #windows path
    X_csv = []
    y_test = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X_csv.append(row[0]) #center image
            y_test.append(row[3]) #steering angle
    print('Testing CSV loaded')
    X_train = [mpimg.imread(imPath) for imPath in X_csv]
    print('Testing images loaded')
    return(X_test, y_test)

def test_function():
    print('Test Passed!')
