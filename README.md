# Ship or Truck Web app 

As the first implementation failed, this is a second iteration with minor differences in the model and the files used

## Project Description
A basic web app that classifies between a ship and a truck

## Learnings
* How to train up basic models and use different tools like LRFinder and EarlyStopping
* Using the Flask Framework to create a web app
* Using Heroku, a Platform As A Service, to host a web application
* Different modules of tensorflow, such as tensorflow-cpu

## Dataset
The dataset used is a basic CIFAR10 dataset which can be used as follows

```python
ds= tf.keras.datasets.cifar10
(x_train_m,y_train),(x_test_m, y_test)=ds.load_data()
```
After this used simple basic pre-processing to take only ship and truck data

```python
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
```

## Frameworks and Algorthims
* Used Tensorflow framework for building the model
* Used Flask to create the web application
* Implemented a basic CNN, trained for 10 epochs
* Used Heroku to deploy the webapp
* Used gunicorn which takes care of everything which happens in etween the web server and the web app

Thanks for going through the ReadME :)
