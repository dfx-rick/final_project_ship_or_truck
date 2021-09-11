import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

ds= tf.keras.datasets.cifar10
(x_train_m,y_train),(x_test_m, y_test)=ds.load_data()

x_train_m=x_train_m/255
x_test= x_test_m/255
y_train= y_train.reshape(-1,)
y_train[:5]

images=np.zeros((10000,32,32,3))#Initializing as numpy arrays
ptr=0
label=np.zeros(10000, dtype='uint8')#Initializing as numpy arrays with similar datatype as y_train
#for loop to go through the images in training data and seperate the images of ship and truck from the rest 
for i in range(len(x_train_m)):
  if(y_train[i]>7):
    images[ptr]=x_train_m[i]
    label[ptr]=y_train[i]-8
    ptr=ptr+1

label=label.reshape(-1,)#reshaping the labels to single dimension

model = models.Sequential([
          layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
          layers.MaxPooling2D((2,2)),
          layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
          layers.MaxPooling2D((2,2)),
          layers.Conv2D(128, (3,3), activation='relu'),
          layers.MaxPooling2D((2,2)),
          layers.Dropout(0.3),
          layers.Flatten(),
          layers.Dense(120, activation='relu'),
          layers.Dense(84, activation='relu'),
          layers.Dense(42, activation='relu'),
          layers.Dense(1, activation='sigmoid')                           
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(images, label, epochs=10,batch_size=512, validation_split=0.1)

model.save('new_model')