import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getData(data_path):
    csv_path = data_path + "driving_log.csv"
    img_path = data_path + "IMG/"

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        csvRows = []
        leftImagePaths = []
        rightImagePaths = []
        centerImagePaths = []
        steeringValue = []
        row = next(reader) #move past the header row
        for row in reader: #format: center name, left name, right name, steering, throttle, brake, speed
            row[0] = data_path + row[0]
            row[1] = data_path + row[1]
            row[2] = data_path + row[2]
            csvRows.append(row)

    #Split into training and testing sets
    csvRows = shuffle(csvRows)
    csvTrain, csvTest = train_test_split(csvRows, test_size = 0.2)
    X_train = [item[0] for item in csvTrain]
    y_train = [item[3] for item in csvTrain]
    X_test = [item[0] for item in csvTest]
    y_test = [item[3] for item in csvTest]
    return(X_train, y_train, X_test, y_test)


def generator_data(data, angle, batch_size):
    index = np.arange(len(data))
    while 1:
        batch_train = np.zeros((batch_size, 160, 320, 3), dtype = np.float32)
        batch_angle = np.zeros((batch_size), dtype = np.float32)
        for i in range(batch_size):
            try:
                random = int(np.random.choice(data.shape[0],1))
            except:
                index = np.arange(len(data))
                batch_train = batch_train[:i,:,:]
                batch_angle = batch_angle[:i]
                break
            batch_train[i] = cv2.imread(data[random])
            batch_train[i] = batch_train[i] - np.mean(batch_train[i])
            batch_angle[i] = angle[random]
        yield (batch_train, batch_angle)

def load_image(image_path):
    img = mpimg.imread(image_path)
    return img

def display_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    print('image size:', img.shape)
