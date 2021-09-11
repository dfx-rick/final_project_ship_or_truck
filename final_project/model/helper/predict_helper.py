import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = tf.keras.models.load_model('saved_model/f_model.h5')

def predict(image, model=model):
    #img = tf.keras.preprocessing.image.load_img(
        #image, target_size=(32, 32,3))
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = tf.expand_dims(img, 0)
    pred = model.predict(img)
    f_pred= np.ceil(pred)
    return pred
