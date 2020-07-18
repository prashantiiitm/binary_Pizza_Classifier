# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:59:39 2020

@author: prashant.pandey
"""

from skimage import io
from skimage.transform import resize
import os
import glob
import pandas as pd,numpy as np
import matplotlib.pyplot as plt
from skimage import color
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
import datetime
from keras.models import model_from_json
from keras.utils import Sequence
import tensorflow as tf

######## Set up the address of the current running script
try:
    filepath = os.path.abspath(os.path.dirname(__file__))
except:
    filepath = "D:/Project Files/Personal/Pizza_Classifier/Codes"
os.chdir(filepath)





training_images_list = glob.glob("..\\Images\\processed\\training\\*.jpg",recursive = True)

img_train = []
label_train = []

for img_id in training_images_list:
    img = io.imread(img_id)
    if(len(img.shape) == 3):
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    img_train.append(img_gray)
    if(img_id.find("\\piz") > 0 ):
        label_train.append([1])
    else:
        label_train.append([0])

img_train = np.array(img_train)
label_train = np.array(label_train)

#################################################
        
        
testing_images_list = glob.glob("..\\Images\\processed\\testing\\*.jpg",recursive = True)

img_test = []
label_test = []

for img_id in testing_images_list:
    img = io.imread(img_id)
    if(len(img.shape) == 3):
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    img_test.append(img_gray)
    if(img_id.find("\\piz") > 0 ):
        label_test.append([1])
    else:
        label_test.append([0])        
        
img_test = np.array(img_test)
label_test = np.array(label_test)


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(validation_split = 0.2)


################################################ create a CNN with 1 Convolutional Layer

img_train = img_train.reshape(16889,200,200,1)
img_test = img_test.reshape(4220,200,200,1)

cnn_model = Sequential()


cnn_model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(200,200,1 )))


cnn_model.add(MaxPool2D(pool_size=(2,2)))

cnn_model.add(Dropout(0.3))

cnn_model.add(Conv2D(10, kernel_size=(3,3), strides=(1,1), activation='relu'))


cnn_model.add(MaxPool2D(pool_size=(2,2)))
# flatten output of conv
cnn_model.add(Flatten())

cnn_model.add(Dropout(0.3))
# output layer
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

cnn_model.summary()


#from numba import cuda
#cuda.select_device(0)
#cuda.close()

a = datetime.datetime.now()
# training the model for 10 epochs
#cnn_model.fit(img_train[:3000,:,:,:], label_train[:3000,:], batch_size=5, epochs=5, validation_split = 0.2)
cnn_model.fit(datagen.flow(img_train,label_train, batch_size = 10),steps_per_epoch=len(img_train) / 32,epochs = 10)

b = datetime.datetime.now()

print(b-a)


#gpus = tf.config.experimental.list_logical_devices('GPU')

scores = cnn_model.evaluate(img_test, label_test)

# serialize model to JSON
model_json = cnn_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_model.save_weights("model.h5")


yhat_classes = cnn_model.predict_classes(img_test, verbose=0)


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(label_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(label_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(label_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(label_test, yhat_classes)
print('F1 score: %f' % f1)

matrix = confusion_matrix(label_test, yhat_classes)
print(matrix)