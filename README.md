# Behavioral Cloning Project

### Overview

In this project,  we use deep neural networks and convolutional neural networks to clone driving behavior. We train, validate and test a model using Keras for outputting a steering angle to an autonomous vehicle.  

We use image data and steering angles which are collected in the Udacity's [simulator](https://github.com/udacity/self-driving-car-sim) to train a neural network and then use this model to drive the car autonomously around the track in simulator.

The main files in this project are: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

### Dependencies
This lab requires [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

**You can setup dependent environment by yourself**

- A  virtual environment install with tensorflow and keras is required
- the tools for running drive.py

```
pip install python-socketio
pip install eventlet
pip install Pillow
pip install flask
```

- data.py

  ```
  pip install -U scikit-learn scipy matplotlib
  ```


### Run

#### drive.py

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and **send the predicted angle back to the server via a websocket connection.**

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

#### video.py

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### Collecting Data

Data collection was done using the simulator in 'Training Mode.' 

 After much trial and error I found that it was not necessary to gather images from a full lap around each track. For the first track I drove just past the bridge to the first dirt patch and for the mountain track I drove only half-way around.  

Using the mouse for steering and being careful to drive steady down the center of the road provided me with good data. The couple of problem areas on each track were addressed by training the vehicle recovering from the sides of the road.  

Successfully driving autonomously around the entire track while only training on a portion of it was an indication to me that the model generalized to all track conditions and did not overfit the training data.  

In the end I gathered over 3500 images from each track for a total data set size of 7650 images.



### Preprocessing Images

Images were preprocessed before feeding to the neural network. 

I tried different color spaces and found that the model generalized best using images in the BGR color space. 

Training with the YUV color space gave erratic steering corrections resulting in too much side-to-side movement of the car.  

In the RGB color space some areas of track 1 required many more training images to navigate correctly, particularly the areas with dirt patches on the side of the road.  

Conversion to BGR solved both of these issues and required less training files overall.  

After conversion to BGR, I crop the unnecessary portions of the image (background of sky, trees, mountains, hood of car) taking 50 pixels off the top of the image and 20 pixels off the bottom.  

The cropped image is then resized to (128, 128) for input into the model.  I found that resizing the images **decreased training time** with no effect on accuracy.



![preprocessing](/media/ubuntu16/%E6%96%B0%E5%8A%A0%E5%8D%B7/Self-Driving/github%E9%A1%B9%E7%9B%AE/CarND-Behavioral-Cloning-master/writeup_images/preprocessing.png)

### Data Augmentation

I tried several methods of data augmentation.  

Images could be augmented by flipping horizontally, 

blurring, 

changing the overall brightness, 

or applying shadows to a portion of the image.  

In the end, adding horizontally flipped images was the only augmentation that was necessary.  

Randomly changing the images to augment the data set was not only was not necessary but when using the RGB color space I saw a loss of accuracy.  

By adding a flipped image for every original image the data set size was effectively doubled. See the functions `random_blur()`, `random_brightness()`, and `random_shadow()` in the file `data.py` for augmentation code. 

Visualization of data augmentation can be found in the Jupyter notebooks `Random_Brightness.ipynb` and `Random_Shadow.ipynb`.

![Augmenting](/media/ubuntu16/%E6%96%B0%E5%8A%A0%E5%8D%B7/Self-Driving/github%E9%A1%B9%E7%9B%AE/CarND-Behavioral-Cloning-master/writeup_images/augmenting.png)

### Data Set Distribution

One improvement that was found to be particularly effective was to fix the poor distribution of the data.  A disproportionate number of steering angles in the data set are at or near zero.  

To correct this steering angles are separated into 25 bins. 

Bins with a count less than the mean are augmented while bins with counts greater than the mean are randomly pruned to bring their counts down. 

This equalizes the distribution of steering angles across all bins.  

Code can be found in the function `fix_distribution()` in `data.py.`  Plotting can be seen in the jupyter notebook `Fix_Distribution.ipynb`

![Distribution](/media/ubuntu16/%E6%96%B0%E5%8A%A0%E5%8D%B7/Self-Driving/github%E9%A1%B9%E7%9B%AE/CarND-Behavioral-Cloning-master/writeup_images/distribution.png)

### 

### Model 

The empirical process of finding the correct neural network can be a lesson in frustration.  I started with LeNet [1] and tried countless modifications.  Feeling the need for something slightly more powerful to increase nonlinearity and work for both tracks I moved on to a modified NVIDIA architecture[2].  Image data is preprocessed as described above before being normalized in the first layer.  The network consists of 5 Convolutional layers with max pooling and 5 fully connected layers.  Dropout layers were used in between the fully connected layers to reduce overfitting.  An Adam optimizer was used with a learning rate of 1e-4. Code for the model can be found in `model_definition.py`

The final network is as follows:

|         Layer          |  Output Shape  | Param # |
| :--------------------: | :------------: | ------: |
| Normalization (Lambda) | (128, 128, 3)  |       0 |
| 1st Convolutional/ReLU | (124, 124, 24) |    1824 |
|      Max Pooling       |  (62, 62, 24)  |       0 |
| 2nd Convolutional/ReLU |  (58, 58, 36)  |   21636 |
|      Max Pooling       |  (29, 29, 36)  |       0 |
| 3rd Convolutional/ReLU |  (25, 25, 48)  |   43248 |
|      Max Pooling       |  (12, 12, 48)  |       0 |
| 4th Convolutional/ReLU |  (10, 10, 64)  |   27712 |
| 5th Convolutional/ReLU |   (8, 8, 64)   |   36928 |
|        Flatten         |     (4096)     |       0 |
|        Dropout         |     (4096)     |       0 |
|  1st Fully Connected   |     (1164)     | 4768908 |
|        Dropout         |     (1164)     |       0 |
|  2nd Fully Connected   |     (100)      |  116500 |
|  3rd Fully Connected   |      (50)      |    5050 |
|  4th Fully Connected   |      (10)      |     510 |
|  5th Fully Connected   |      (1)       |      11 |

### Training

------

Training the network is done using the python script `model.py`.  By default the script runs for 10 epochs although the script will take a different number of epochs as a parameter.  The script by default allocates 80% of the data set to training and 20% to validation sets.

```
python model.py 20  # train for 20 epochs.
```

The script reads the image data from the CSV file, performs any augmentation and distribution balancing, and starts training on the data set.  Due to the possibility that the entire collection of image files might not fit into memory all at once the data set is separated into batches by a Keras generator.  The generator creates batches of `BATCH_SIZE` images, reading image data and augmenting the images on the fly. See the function `get_generator()`, line 383, of the file `data.py` for generator code.  Generators were used for both training and validation sets.

Validation accuracy was not a good indicator of the performance of the network.  It was better to watch for overfitting by comparing the training mean squared error (MSE) with the validation MSE.  A climbing validation MSE while the training MSE was still decreasing was a sign that the model was overfitting.  Dropout was used to combat this but I also limited the number of epochs to get the best performing model.

### Videos

`video1.mp4` is the recording of the vehicle going around the basic track.  `video2.mp4` is the recording of the vehicle going around the mountain track.







#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains **dropout layers** in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Tips

- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.