class Modified_Nvidia_Netwrok:
    def __init__(self, save_path):
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Conv2D, Lambda, Dropout
        from keras.layers.pooling import MaxPooling2D
        from keras import optimizers
        from keras.callbacks import ModelCheckpoint
        ## 
        from keras.layers import Cropping2D, BatchNormalization, Activation
        from keras.layers.advanced_activations import ELU
        from keras.regularizers import l2

        self.model = Sequential()

        # Normalization (None, 128, 128, 3)
        self.model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(160, 320, 3)))

        # Convolutional Layer (None, 124, 124, 24)
        self.model.add(Conv2D(24, kernel_size=(5,5), padding='valid', activation='relu'))
        # Maxpooling Layer (None, 62, 62, 24)
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        # (None, 58, 58, 36) -> (None, 29, 29, 36)
        self.model.add(Conv2D(36, kernel_size=(5,5), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        # (None, 25, 25, 48) -> (None, 12, 12, 48)
        self.model.add(Conv2D(48, kernel_size=(5,5), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        # (None, 10, 10, 64)
        self.model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
        
        # (None, 8, 8, 64)
        self.model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))

        # Flattening Layer (None, 4096)
        self.model.add(Flatten())

        # Dropout 0.5 (None, 4096)
        self.model.add(Dropout(0.5))

        # Fully Connected (None, 1164)
        self.model.add(Dense(1164, activation='relu'))
        
        # 
        self.model.add(Dropout(0.5))

        # (None, 100)
        self.model.add(Dense(100, activation='relu'))

        # (None, 50)
        self.model.add(Dense(50, activation='relu'))

        # (None, 10)
        self.model.add(Dense(10, activation='relu'))

        # (None, 1)
        self.model.add(Dense(1, kernel_initializer='normal'))

        
        ## optimizer
        optimizer = optimizers.Adam(lr = 0.0001)

        self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.model.summary()
        # Use the keras ModelCheckpoint to save the model 
        # afer every epoch
        model_checkpoint = ModelCheckpoint(save_path, save_best_only=True)
        self.callbacks = [model_checkpoint]


    ## Train the model using Keras' fit_generator()
    
    # train_generator: generator to provide batches of training data
    # valid_generator: generator to provide batches of validation data
    # training_steps: integer of training steps to achieve one epoch
    # validation_steps: integer of validation steps
    # epochs: integer

    def fit(self, train_generator, valid_generator, training_steps, validation_steps, epochs=10):

        print("Training with {} training steps, {} validation steps.".format(training_steps, validation_steps))

        self.model.fit_generator(train_generator,
                                 steps_per_epoch = training_steps,
                                 validation_data = valid_generator,
                                 validation_steps = validation_steps,
                                 epochs = epochs,
                                 callbacks = self.callbacks)

'''
import os
import csv
import cv2
import numpy as np 

from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense

## reading data
lines = []
dataset_dir = '/media/ubuntu16/新加卷/Self-Driving/datasets/carnd'
data_file = os.path.join(dataset_dir, 'driving_log.csv')

with open(data_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

center_images = []
steer_labels = []

for line in lines:
    filepath = line[0]
    center_image = cv2.imread(filepath)
    center_images.append(center_image)
    steer_label = float(line[3])
    steer_labels.append(steer_label)

X_train = np.array(center_images)
y_train = np.array(steer_labels)


## simple network

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

# save model
model.save('simple_model.h5')

Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', 
data_format=None, dilation_rate=(1, 1), activation=None, 
use_bias=True, kernel_initializer='glorot_uniform', 
bias_initializer='zeros', kernel_regularizer=None, 
bias_regularizer=None, activity_regularizer=None, 
kernel_constraint=None, bias_constraint=None)
'''