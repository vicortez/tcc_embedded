import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet,vgg19
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

 
#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')

vgg19=vgg19.VGG19(weights='imagenet')
 
#Load the Inception_V3 model
#inception_model = inception_v3.InceptionV3(weights='imagenet')
 
#Load the ResNet50 model
#resnet_model = resnet50.ResNet50(weights=)
 
#Load the MobileNet model
#mobilenet_model = mobilenet.MobileNet(weights='imagenet')

model=vgg19
def classify(img):
    #old_size = img.shape[:2] # old_size is in (height, width) format
    #desired_size=224
    #ratio = float(desired_size)/max(old_size)
    #new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    #img = cv2.resize(img, (new_size[1], new_size[0]))
    
    #delta_w = desired_size - new_size[1]
    #delta_h = desired_size - new_size[0]
    #top, bottom = delta_h//2, delta_h-(delta_h//2)
    #left, right = delta_w//2, delta_w-(delta_w//2)
    
    #color = [0, 0, 0]
    #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    img=cv2.resize(img,(224,224))

    #print('PIL image size',img.size)
     
    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(img)
    plt.imshow(np.uint8(numpy_image))
    plt.show()
    print('numpy array size',numpy_image.shape)
    
     
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    print('image batch size', image_batch.shape)
    plt.imshow(np.uint8(image_batch[0]))
    
    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch.copy())
     
    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    # print predictions
     
    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    label = decode_predictions(predictions)
    print (label)
