import os
import csv
import cv2
import numpy as np 

from keras.models import Sequential
from keras.layers import Flatten, Dense

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


## save model
model.save('simple_model.h5')