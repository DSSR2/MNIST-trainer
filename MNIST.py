import numpy as np
import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import pandas as pd


seed = 10
np.random.seed(seed)
num_pixels = 784
num_classes = 10
global CONST_DATA
CONST_DATA = pd.read_csv('train.csv')

def modData(wri, data):
    data = data.drop(wri).reset_index(drop = True)
    return data

def findWrong(pred, valid):
    pred = [i.argmax() for i in pred]
    valid = [i.argmax() for i in valid]
    wrong = []
    wrongIndex = []
    for i in range(len(pred)):
        if(pred[i] != valid[i]):
            wrong.append((i, pred[i], valid[i]))
            wrongIndex.append(i)
    return wrong, wrongIndex

def test(data):
    Y = data.label
    data.drop(['label'], axis = 1, inplace = True)
    data = data.values.reshape(data.shape[0], 1, 28, 28)
    Y = np_utils.to_categorical(Y)
    pred = model.predict(data)

    [wr, wri] = findWrong(pred, Y)
    print('WRONG COUNT: ', len(wri))
    return wri
    
def saveModel():
    model_json = model.to_json()
    with open('model.json', 'w') as jf:
        jf.write(model_json)
    model.save_weights('model.h5')


def splitData(data, splitAmt = 10):
    validationRows = []
    n = (int)((splitAmt/100)*data.shape[0])
    for i in range(n):
        num = random.randint(0, data.shape[0]-1)
        if(num not in validationRows):
            validationRows.append(num)
        
    data_validation = data.ix[validationRows].reset_index(drop=True)
    data = data.drop(validationRows).reset_index(drop=True)

    Y_train = data.label
    Y_validation = data_validation.label

    data.drop(['label'], axis = 1, inplace = True)
    data_validation.drop(['label'], axis = 1, inplace = True)

    return data, Y_train, data_validation, Y_validation 

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def makeModel():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
    
def train(data):
    [X_train, Y_train, X_validation, Y_validation] = splitData(data_train, 10)
    X_train = X_train.values.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_validation = X_validation.values.reshape(X_validation.shape[0], 1, 28, 28).astype('float32')

    X_train = X_train/255
    X_validation = X_validation/255

    # one hot encode outputs
    Y_train = np_utils.to_categorical(Y_train)
    Y_validation = np_utils.to_categorical(Y_validation)
    num_classes = Y_train.shape[1]

    global model
    model =  makeModel()
    model.fit(X_train, Y_train, validation_data = (X_validation, Y_validation), epochs = 10, batch_size = 150)
    #model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=5, batch_size=200, verbose=2)
    
noOfRep = 4
for i in range(noOfRep):
    
    data_train = CONST_DATA.copy()
    train(data_train)
    wri = test(data_train)
    CONST_DATA = modData(wri, CONST_DATA).copy()
saveModel()
