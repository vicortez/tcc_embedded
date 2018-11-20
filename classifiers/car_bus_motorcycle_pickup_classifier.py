from keras.datasets import cifar100 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import os
from scipy import misc
import matplotlib.pyplot as plt
import cv2

def classify(img):
    img=cv2.resize(img,(224,224))
    plt.imshow(np.uint8(img))
    plt.show()
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    img=np.array([img])
     
    # get the predicted probabilities for each class
    predictions = model.predict(img)
    print(predictions,LABELS[LABELS.index(max(predictions))])


CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

LABELS = ['bus', 'motorcycle', 'pickup_truck', 'streetcar']
LABELS.index('bus')

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 8 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels
pool_size = 2 # we will use 2x2 pooling
conv_depth_1 = 20 # we will initially have 32 kernels per conv. layer.
conv_depth_2 = 50 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

(x_train, y_train), (x_test, y_test) = cifar100.load_data() # fetch CIFAR-10 data

filtered_x_train=[]
filtered_y_train=[]
filtered_x_test=[]
filtered_y_test=[]

for pos,i in enumerate(y_train):
    #bus
    if i==13:
        filtered_x_train.append(x_train[pos])
        filtered_y_train.append(0)
    #motorcycle
    elif i==48:
        filtered_x_train.append(x_train[pos])
        filtered_y_train.append(1)
    #pickup_truck
    elif i==58:
        filtered_x_train.append(x_train[pos])
        filtered_y_train.append(2)
    #streetcar
    elif i==81:
        filtered_x_train.append(x_train[pos])
        filtered_y_train.append(3)

for pos,i in enumerate(y_test):
    #bus
    if i==13:
        filtered_x_test.append(x_test[pos])
        filtered_y_test.append(0)
    #motorcycle
    elif i==48:
        filtered_x_test.append(x_test[pos])
        filtered_y_test.append(1)
    #pickup_truck
    elif i==58:
        filtered_x_test.append(x_test[pos])
        filtered_y_test.append(2)
    #streetcar
    elif i==81:
        filtered_x_test.append(x_test[pos])
        filtered_y_test.append(3)

del filtered_x_test[22]
del filtered_x_test[31]
del filtered_x_test[57]
del filtered_x_test[104]
del filtered_x_test[342]
del filtered_x_test[343]
del filtered_x_test[348]
del filtered_y_test[22]
del filtered_y_test[31]
del filtered_y_test[57]
del filtered_y_test[104]
del filtered_y_test[342]
del filtered_y_test[343]
del filtered_y_test[348]


del filtered_x_train[6]
del filtered_x_train[17]
del filtered_x_train[41]
del filtered_x_train[98]
del filtered_x_train[133]
del filtered_x_train[137]
del filtered_x_train[161]
del filtered_x_train[192]
del filtered_x_train[207]
del filtered_y_train[6]
del filtered_y_train[17]
del filtered_y_train[41]
del filtered_y_train[98]
del filtered_y_train[133]
del filtered_y_train[137]
del filtered_y_train[161]
del filtered_y_train[192]
del filtered_y_train[207]

filtered_x_train=np.array(filtered_x_train)
filtered_y_train=np.array(filtered_y_train)
filtered_x_test=np.array(filtered_x_test)
filtered_y_test=np.array(filtered_y_test)
num_train, height, width, depth = filtered_x_train.shape # there are 50000 training examples in CIFAR-10
num_test = filtered_x_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(filtered_y_train).shape[0] # there are 10 image classes

filtered_x_train = filtered_x_train.astype('float32')
filtered_x_test = filtered_x_test.astype('float32')
filtered_x_train /= np.max(filtered_x_train) # Normalise data to [0, 1] range
filtered_x_test /= np.max(filtered_x_test) # Normalise data to [0, 1] range

filtered_Y_train = np_utils.to_categorical(filtered_y_train, num_classes) # One-hot encode the labels
filtered_Y_test = np_utils.to_categorical(filtered_y_test, num_classes) # One-hot encode the labels







inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(filtered_x_train, filtered_Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
scores = model.evaluate(filtered_x_test, filtered_Y_test, verbose=1)  # Evaluate the trained model on the test set!
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Saved model to disk")
with open("modelresult.txt","w") as txt:
    s="this model has achieved an acurracy of " , scores[1] , " with 5000 testing samples and ",num_epochs," epochs"
    txt.write(str(s))