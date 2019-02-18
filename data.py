import os
import csv
import cv2
import copy
import random
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

STEERING_CORRECTION_LEFT  = 0.2
STEERING_CORRECTION_RIGHT = 0.2
USE_SIDE_CAMERAS = True 
FLIP_IMAGES = True

## self defined data structure
class SteeringData:
    def __init__(self, image_path, steer, flipped_flag):
        self.image_path = image_path
        self.steer = steer
        self.flipped_flag = flipped_flag
        self.shadow_flag = 0
        self.bright_flag = 0
        self.blur_flag = 0


## Reading data from csv file 
## saving it as the self defined data structure
def csv_read(csv_path):
    print("Reading data from csv file...")
    data = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)

        for line in reader:
            center_image_path = line[0]
            if os.path.isfile(center_image_path):
                center_steer = float(line[3])
                data.append(SteeringData(center_image_path, center_steer, 0))
                if FLIP_IMAGES:
                    data.append(SteeringData(center_image_path, -center_steer,1)) 

            if USE_SIDE_CAMERAS:
                left_image_path = line[1]
                if os.path.isfile(left_image_path):
                    left_steer = center_steer + STEERING_CORRECTION_LEFT
                    data.append(SteeringData(left_image_path, left_steer, 0))
                    if FLIP_IMAGES:
                        data.append(SteeringData(left_image_path, -left_steer, 1))
                right_image_path = line[2]
                if os.path.isfile(right_image_path):
                    right_steer = center_steer - STEERING_CORRECTION_RIGHT
                    data.append(SteeringData(right_image_path, right_steer, 0))
                    if FLIP_IMAGES:
                        data.append(SteeringData(right_image_path, -right_steer, 1))
    print("Reading is done.")

    return shuffle(data)


## return training and validation data sets
def load_data_sets(csv_path, split=0.2):
    data = csv_read(csv_path)
    train, valid = train_test_split(data, test_size=split)

    return train, valid
    

def get_bin_counts(x):
    steers = [item.steer for item in x]

    bin_count = 25
    max_bin = np.max(steers)
    min_bin = np.min(steers)
    spread = max_bin - min_bin
    bin_size = spread / bin_count

    bins = [min_bin + i*bin_size for i in range(bin_count)]
    bins.append(max_bin + 0.1)

    hist, bin_edges = np.histogram(steers, bins)
    
    # desired_count_per_bin = int(np.mean(bin_counts)) * 2
    desired_per_bin = int(np.mean(hist)*1)

    return bin_edges, hist, desired_per_bin


# This method takes the dataset supplied by x
# and adds images.
# Existing images in the dataset are copied
# and augmented by a random blur, random
# shadows, and/or random brightness changes.

def augment_dataset(x, fix_dist):
    bin_edges, hist, desired_per_bin = get_bin_counts(x)

    copy_times = np.float32((desired_per_bin-hist)/hist)
    copy_times_accum = np.zeros_like(copy_times)

    augmented = []
    for i in range(len(x)):
        data = x[i]
        
        index = np.digitize(data.steer, bin_edges) -1
        copy_times_accum[index] += copy_times[index]
        copy_times_integer = np.int32(copy_times_accum[index])
        copy_times_accum[index] -= copy_times_integer
        
        for j in range(copy_times_integer):
            new_data = copy.deepcopy(data)
            new_data.shadow_flag = int(np.random.uniform(0,1) + 0.5)
            new_data.blur_flag   = int(np.random.uniform(0,1) + 0.5)
            new_data.bright_flag = int(np.random.uniform(0,1) + 0.5)
            augmented.append(new_data)

    if (fix_dist):
        return fix_distribution(x + augmented, bin_edges, hist, desired_per_bin)
    else:
        return x + augmented

## 
def fix_distribution(x, bin_edges, hist, desired_per_bin):
    # ensure we don't divide by zero
    non_zero_hist = np.array(hist)
    non_zero_hist[non_zero_hist==0] = desired_per_bin

    keep_percentage = np.float32(desired_per_bin/non_zero_hist)

    def should_keep(item):
        prob_to_keep = keep_percentage[np.digitize(item.steer, bin_edges)-1]

        random_prob = np.random.uniform(0,1)

        return (random_prob <= prob_to_keep)

    trimmed_training_set = [item for item in training_set if should_keep(item)]

    return trimmed_training_set

## 
def preprocess_image(img):
    # img coming in is an RGB image of shape: (160, 320, 3).  
    # Convert to BGR, (the trained network processinh BGR image)
    # Crop 50 pixels off the top of the image, and 20 pixels off the bottom. 
    # Then resize to 128, 128, 3
    
    ## ??
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cropped = bgr[50:140, :] # height, width
    #resized = cv2.resize(cropped, (128,128))
    resized = cv2.resize(cropped, (320,160)) # width, height

    return resized

## 
def random_blur(image):
      #
    # Generate a random odd number for our
    # kernel size between 3 and 9
    #
    kernel_size = (np.random.randint(1, 5) * 2) + 1

    #
    # Blur and return
    #
    return cv2.GaussianBlur(image, (kernel_size, kernel_size),  0)

## 
def random_shadow(image):
    height, width = image.shape[:2]

    number_of_shadows = np.random.randint(1, 6)
    list_of_shadows = []

    # define every shadow by randomly determine 
    # the number and positions of a polygon's vertices
    for i in range(number_of_shadows):
        shadow_vertices = []
    
        number_of_vertices = np.random.randint(3,26)
    
        for j in range(number_of_vertices):
            position_x = width * np.random.uniform()
            position_y = height * np.random.uniform()
        
            shadow_vertices.append((position_x,position_y))
    
        list_of_shadows.append(np.array([shadow_vertices], dtype=np.int32))
        
    # create a mask with the same dimensions as the original image
    mask = np.zeros((height,width))

    # fill all the shadow polygon
    for shadow in list_of_shadows:
        cv2.fillPoly(mask, shadow, 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]

    # randomly choose a drakness of the shadows
    # lower numbers result in darker shadows

    random_darkness = np.random.randint(45, 75) /100.0

    v_channel[mask==255] = v_channel[mask==255]*random_darkness

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr 

## 
def random_brightness(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = image_hsv[:,:,2]

    # get a random number to represent the change in brightness
    # note that all the v channel values should add a same number
    ''' 
    int8: -128~127, int16: -32768~32767, uint8: 0~255
    differnt types of data can not operate with each other 
    '''
    brightness_change = np.random.randint(0,100,dtype=np.uint8)

    # apple brightness change
    # be sure that v_channel value will not less 0 or great 255 while adding or substracting !
    if (np.random.uniform(0,1) > 0.5):
        v_channel[v_channel>(255-brightness_change)] = 255
        v_channel[v_channel<=(255-brightness_change)] += brightness_change
    
    else:
        v_channel[v_channel<(brightness_change)] = 0
        v_channel[v_channel>=(brightness_change)] -= brightness_change
    # using v[v >= make_positive] + brightness_change can avoid this problem!

    # put the changed v channel back to hsv and covert to rgb
    # this line can be deleted becase that v_channel is the reference of image_hsv[]
    image_hsv[:,:,2] = v_channel
    
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)



## process images by their flags 
def get_generator(images, batch_size):
    while True:
        # grap a random sample of size "batch_size"
        # from the "images" array

        batch = np.random.choice(a=images, size=batch_size)

        X = []
        y = []

        for index in range(len(batch)):
            image_data = batch[index]

            image_path = image_data.image_path

            if os.path.isfile(image_path):
                # Read the image, apply data augmentation
                # and add to the batch

                image = cv2.imread(image_path)
                steer = image_data.steer

                if image is not None:
                    if image_data.flipped_flag == 1:
                        image = cv2.flip(image, 1)
                    
                    if image_data.blur_flag == 1:
                        image = random_blur(image)
                    
                    if image_data.bright_flag == 1:
                        image = random_brightness(image)

                    if image_data.shadow_flag == 1:
                        image = random_shadow(image)

                    image = preprocess_image(image)

                    X.append(image)
                    y.append(steer)

        # covert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        yield shuffle(X, y)