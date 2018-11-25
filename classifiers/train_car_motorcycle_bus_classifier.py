from keras.datasets import cifar100 # subroutines for fetching the CIFAR-100 dataset
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import os
from scipy import misc
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.metrics import confusion_matrix
import itertools

def classify(img):
    img=cv2.resize(img,(32,32))
    print("shape:", img.shape)
    plt.imshow(img)
    plt.show()
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    img=np.array([img])
    print("shape:", img.shape)
    # get the predicted probabilities for each class
    predictions = model.predict(img)
    print(predictions,LABELS[np.argmax(predictions[0])])
    
def showsome(n=10,set='test'):
    if set=='train':
        for el in random.sample(range(1, len(filtered_x_train)-1), n):
            plt.imshow(filtered_x_train[el])
            plt.show()
            print(filtered_y_train[el])
        
    elif set=='test':
        for el in random.sample(range(1, len(filtered_x_test)-1), n):
            plt.imshow(filtered_x_test[el])
            plt.show()
            print(filtered_y_test[el])
            
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
          
def model_conf():
    y_pred = [np.argmax(el) for el in model.predict(filtered_X_test)]
    y_true = filtered_y_test
    cnf_matrix = confusion_matrix(y_true,y_pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=LABELS,
                      title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=LABELS, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()
    

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

SEED = 42
LABELS = ['bus', 'motorcycle', 'streetcar']
LABELS.index('bus')

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 32 # we iterate x times over the entire training set
kernel_size = 3 # we will use 3x3 kernels
pool_size = 2 # we will use 2x2 pooling
conv_depth_1 = 20 # we will initially have 32 kernels per conv. layer.
conv_depth_2 = 50 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons


# constructing and shaping dataset ===
(x_train, y_train), (x_test, y_test) = cifar100.load_data() # fetch CIFAR-100 data
(x_train2, y_train2), (x_test2, y_test2) = cifar10.load_data() # fetch CIFAR-10 data

filtered_x_train=[]
filtered_y_train=[]
filtered_x_test=[]
filtered_y_test=[]

cars_x_train=[]
cars_y_train=[]
cars_x_test=[]
cars_y_test=[]

for pos,i in enumerate(y_train):
    #bus = 0
    if i==13:
        filtered_x_train.append(x_train[pos])
        filtered_y_train.append(0)
    #motorcycle = 1
    elif i==48:
        filtered_x_train.append(x_train[pos])
        filtered_y_train.append(1)

for pos,i in enumerate(y_test):
    #bus
    if i==13:
        filtered_x_test.append(x_test[pos])
        filtered_y_test.append(0)
    #motorcycle
    elif i==48:
        filtered_x_test.append(x_test[pos])
        filtered_y_test.append(1)
        
#getting the cars from the cifar10 dataset    
for pos,i in enumerate(y_train2):
    #car = 2
    if i==1:
        cars_x_train.append(x_train2[pos])
        cars_y_train.append(2)

for pos,i in enumerate(y_test2):
    #car
    if i==1:
        cars_x_test.append(x_test2[pos])
        cars_y_test.append(2)
        
for el in random.sample(range(1, len(cars_x_train)-1), 600):
    pos = random.randint(1,len(filtered_x_train)-1)
    filtered_x_train.insert(pos ,cars_x_train[el])
    filtered_y_train.insert(pos ,cars_y_train[el])
       
for el in random.sample(range(1, len(cars_x_test)-1), 120):
    pos = random.randint(1,len(filtered_x_test)-1)
    filtered_x_test.insert(pos ,cars_x_test[el])
    filtered_y_test.insert(pos ,cars_y_test[el])
    

            

filtered_x_train=np.array(filtered_x_train)
filtered_y_train=np.array(filtered_y_train)
filtered_x_test=np.array(filtered_x_test)
filtered_y_test=np.array(filtered_y_test)
num_train, height, width, depth = filtered_x_train.shape # there are 50000 training examples in CIFAR-10
num_test = filtered_x_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(filtered_y_train).shape[0] # there are 10 image classes

filtered_X_train = filtered_x_train.astype('float32')
filtered_X_test = filtered_x_test.astype('float32')
filtered_X_train /= np.max(filtered_x_train) # Normalise data to [0, 1] range
filtered_X_test /= np.max(filtered_x_test) # Normalise data to [0, 1] range

filtered_Y_train = np_utils.to_categorical(filtered_y_train, num_classes) # One-hot encode the labels
filtered_Y_test = np_utils.to_categorical(filtered_y_test, num_classes) # One-hot encode the labels
val = (filtered_X_test,filtered_Y_test)

# model ===
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

model.fit(filtered_X_train, filtered_Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0,validation_data=val) # ...holding out 10% of the data for validation
scores = model.evaluate(filtered_X_test, filtered_Y_test, verbose=1)  # Evaluate the trained model on the test set!
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model_tcc.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_tcc_weights.h5")
print("Saved model to disk")
with open("model_tcc_result.txt","w") as txt:
    s="this model has achieved an acurracy of " , scores[1] , " with 5000 testing samples and ",num_epochs," epochs"
    txt.write(str(s))