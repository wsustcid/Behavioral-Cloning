# End-to-End Driving via Imitation Learning

## 1. Introduction
<div align=center><img src=./doc/assets/cover.png /></div>

In this project,  we use deep neural networks and convolutional neural networks to clone the human driver's driving behavior. The trained end-to-end driving model outputs steering angles to keep the car driving within the track.  

## 2. Requirements
  - Ubuntu 16.04
  - Virtual environment with python 2.7 
  - Tensorflow 
  - Keras

**Installation**
```python
pip install opencv-python
pip install -U scikit-learn scipy matplotlib
# the tools for running drive.py
pip install python-socketio
pip install eventlet
pip install Pillow
pip install flask
```
  - This lab requires [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) 
  - The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## 3. Data
  - The driving images and steering angles are collected in the Udacity's [simulator](https://github.com/udacity/self-driving-car-sim)

<div align=center><img src=./doc/assets/simulator.png width=500/></div>

## 4. Train & Run
```python
# train and save model
python train.py 

#Once the model has been saved, it can be used with drive.py using this command:
python drive.py model.h5
```
  - `drive.py` will load the trained model and use the model to make predictions on individual images in real-time and **send the predicted angle back to the server via a websocket connection.**

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

## 5. Saving a video of the autonomous agent

```python
# save image
python drive.py model.h5 run1
# crate video
python video.py run1
# optionally
python video.py run1 --fps 48 # The default FPS is 60.

```
 - The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.


## 6. License
This Project is released under the [Apache](LICENSE) licenes.