# -*- coding: utf-8 -*-

import keras
import numpy as np
import cv2, Image

import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing import image
from keras.applications import imagenet_utils, mobilenet


def prepare_image(img):
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    img=np.array([img])
    expanded_img = np.expand_dims(img,axis=0)
    print(expanded_img.shape())
    return keras.applications.mobilenet.preprocess_input(expanded_img)
def process_image(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = Image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  pImg = mobilenet.preprocess_input(img_array)
  return pImg

def classify(img):
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pil_im = image.fromarray(cv2_im)
    img_array = pil_im.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pImg = mobilenet.preprocess_input(img_array)
    
    prediction = mobilenet.predict(pImg)
    # obtain the top-5 predictions
    results = imagenet_utils.decode_predictions(prediction)
    print(results)
    
#img =cv2.imread('gol.jpg')
#img=cv2.resize(img,(224,224))
#plt.imshow(np.uint8(img))
#plt.show()
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
#img=np.array([img])
#mobile = keras.applications.mobilenet.MobileNet()
    # get the predicted probabilities for each class
#predictions = mobile.predict(img)
#results = imagenet_utils.decode_predictions(predictions)
#print(results)
    
#mobile = keras.applications.mobilenet.MobileNet()
#preprocessed_img = prepare_image(cv2.imread('gol.jpg'))
#plt.imshow(np.uint8(preprocessed_img))
#predictions = mobile.predict(preprocessed_img)
#results = imagenet_utils.decode_predictions(predictions)
#print(results)
    
 # process the test image
#pImg = process_image('examplecar.png')
  # define the mobilenet model
#mobilenet = mobilenet.MobileNet()
  # make predictions on test image using mobilenet
#prediction = mobilenet.predict(pImg)
  # obtain the top-5 predictions
#results = imagenet_utils.decode_predictions(prediction)
#print(results)