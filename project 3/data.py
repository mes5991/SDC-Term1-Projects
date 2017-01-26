import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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
            row[0] = img_path + row[0]
            row[1] = img_path + row[1]
            row[2] = img_path + row[2]
            csvRows.append(row)

    #Split into training and testing sets
    csvRows = shuffle(csvRows)
    csvTrain, csvTest = train_test_split(csvRows, test_size = 0.2)
    X_train = [item[0] for item in csvTrain]
    y_train = [item[3] for item in csvTrain]
    X_test = [item[0] for item in csvTest]
    y_test = [item[3] for item in csvTest]
    return(X_train, y_train, X_test, y_test)

# X_train, y_train, X_test, y_test = getData("data/udacityData/data/")
