from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def classify(img):
    img = cv2.resize(img, (32, 32)).astype(np.float32)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(np.uint8(img_rgb))
    plt.show()
    print(img.shape)
    img_rgb /= 255
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    img_array=np.array([img_rgb])
    # get the predicted probabilities for each class
    with graph.as_default():
        predictions = loaded_model.predict(img_array)
        print(predictions,LABELS[np.argmax(predictions[0])])

    
LABELS = ['motorcycle', 'streetcar']
    
# load json and create model
json_file = open('classifiers/model_tcc.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classifiers/model_tcc_weights.h5")
print("Loaded model from disk")
graph = tf.get_default_graph()