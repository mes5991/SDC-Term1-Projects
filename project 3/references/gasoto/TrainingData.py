import cv2, numpy as np


def getData(data,angle,batch_size):
    index = np.arange(len(data))
    while 1:        
        batch_train = np.zeros((batch_size, 40, 160, 3), dtype = np.float32)
        batch_angle = np.zeros((batch_size,), dtype = np.float32)        
        
        for i in range(batch_size):
            try:
                random = int(np.random.choice(index,1,replace = False))
            except :
                index = np.arange(len(data))    
                batch_train = batch_train[:i,:,:]
                batch_angle = batch_angle[:i]
                break
            batch_train[i] = cv2.imread(data[random])
            batch_train[i] = batch_train[i] - np.mean(batch_train[i])
            batch_angle[i] = angle[random]
        yield (batch_train, batch_angle)
