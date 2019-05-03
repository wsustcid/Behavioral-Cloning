class: middle, center

#Behavioral Cloning Project

####Shuai Wang

####USTC, March 12, 2019

Copyright (c) 2019 Shuai Wang. All rights reserved.



---



# 1. Overview

## 1.1 Behavioral Cloning

***Synonyms:***

- Apprenticeship learning; 
- Behavioral cloning; 
- Learning by demonstration; 
- Learning by imitation;
- Learning control rules

.footnote[*Ref. Encyclopedia of Machine Learning*]

### 1.1.1 Defination

Behavioral cloning is a method by which human sub-cognitive skills can be captured and reproduced in a
computer program. 

- As the human subject performs the skill, his or her actions are recorded along with the situation that gave rise to the action. 
- A log of these records is used as input to a learning program. 
- The learning program outputs a set of rules that reproduce the skilled behavior. 

This method can be used to construct automatic control systems for complex tasks for which classical control theory is inadequate. It can also be used for training.



### 1.1.2 Motivation and Background

Behavioral cloning (Michie, Bain, & Hayes-Michie, 1990) is a form of **learning by imitation** whose main
motivation is to build a model of the behavior of a human when performing a complex skill. 

Experts might be defined as people who know what they are doing not what they are talking about. 

- That is, once a person becomes highly skilled in some task, the skill becomes sub-cognitive and is no longer available to introspection. So when the person is asked to explain why certain decisions were made, the explanation is a *post hoc* justification rather than a true explanation. 

*Preferably, the model should be in a readable form. It is related to other forms of learning by imitation, such as **inverse reinforcement learning** (Abbeel & Ng, 2004; Amit & Matarić, 2002; Hayes & Demiris, 1994;
Kuniyoshi, Inaba, & Inoue, 1994; Pomerleau, 1989) and methods that use data from human performances to model the system being controlled (Atkeson & Schaal, 1997; Bagnell & Schneider, 2001).*



### 1.1.3 Structure of the Learning System

Behavioral cloning assumes that there is a plant of some kind that is under the control of a human operator. The plant may be a physical system or a simulation. 

- In either case, the plant must be instrumented so that it is possible to capture the state of the system, including all the control settings. 
- Thus, whenever the operator performs an action, that is, changes a control setting, **we can associate that action with a particular state**.

.center[<img src=imgs/p0_1.png height=350px/>]



---

Let us use a simple example of a system that has only one control action. 

- A pole balancer has four state variables: the angle of the pole, $\theta$, and its angular velocity, $\dot{\theta}$and the position, $x$, and velocity $\dot{x}$, of the cart on the track. The only action available to the controller is to apply a fixed positive or negative force, $F$, to accelerate the cart left or right.
- Whenever a control action is performed, we record the action as well as values of the four state variables at the time of the action. **Each of these records can be viewed as an example of a mapping from state to action.**

.center[<img src=imgs/pole_balancer.gif />]



---

#### 1.1.3.1 Learning Direct (Situation–Action) Controllers

A controller such as the one described above is referred to as a ***direct controller*** because it maps **situations** to **actions**.

**Limitations:** 

Direct controllers work quite well for systems that have a relatively small state space. However, for complex systems, behavioral cloning of direct situation–action rules tends to produce very brittle controllers. That is, they cannot tolerate large disturbances. 

- For example, when air turbulence is introduced into the flight simulator, the performance of the clone degrades very rapidly. This is because the examples provided by logging the performance of a human only cover a very small part of the state space of a complex system such as an aircraft in flight. Thus, the“expertise” of the controller is very limited. If the system strays outside the controller’s region of expertise, **it has no method for recovering and failure is usually catastrophic.**

More robust control is possible but only with a significant change in approach. The more successful methods decompose the learning task into two stages: 

- learning goals 
- and learning the actions to achieve those goals.

---

***Example:***

.center[<img src=imgs/airplane-controls.gif />]

A pilot of a fixed-wing aircraft can control the ailerons(副翼), elevators(升降舵), rudder(方向舵), throttle, and flaps(襟翼). To build an autopilot, the learner must build a system that can set each of the control variables. Sammut et al. (1992), viewed this as ***a multitask learning problem***.

------

**Solutions to multi-variable control:**

- Each training example is a feature vector that includes the position, orientation, and velocities of the
  aircraft as well as the values of each of the control settings: ailerons, elevator, throttle, and flaps. The rudder is ignored. 
- A separate decision tree is built for each control variable. For example, the aileron setting is treated as the dependent variable and **all the other variables, including the other controls**, are treated as the attributes of the training example. A decision tree is built for ailerons, then the process is repeated for the elevators, etc. 
- The result is a decision tree for each control variable.
- The autopilot code executes each decision tree in each cycle of the control loop. 

This method treats the setting of each control as a separate task. It may be surprising that this method works since it is often necessary to adjust more than one control simultaneously to achieve the desired result. For example, to turn, it is normal to use the ailerons to roll the aircraft while adjusting the elevators to pull it around. This kind of multivariable control does result from multiple decision trees. When, say, the aileron decision tree initiates a roll, the elevator’s decision tree detects the roll and causes the aircraft to pitch up and execute a turn.

------



#### 1.1.3.2 Learning Indirect (Goal-Directed) Controllers

The problem of learning in a large search space can partially be addressed by decomposing the learning into subtasks. A controller built in this way is said to be an ***indirect controller***. 

- A control is “indirect” if it does not compute the next action directly from the system’s current state but uses, in addition, some intermediate information. 
- An example of such intermediate information is a subgoal to be attained before achieving the final goal.

---

***Steps:***

1. Subgoals often feature in an operator’s control strategies and can be automatically detected from a
   trace of the operator’s behavior (Šuc & Bratko, 1997). 
   - The problem of subgoal identification can be treated as the inverse of the usual problem of controller design, that is, given the actions in an operator’s trace, find the goal that these actions achieve. 
   - The limitation of this approach is that it only works well for cases in which there are just a few subgoals, not when the operator’s trajectory contains many subgoals. 
2. In these cases, a better approach is to generalize the operator’s trajectory. The generalized trajectory can be viewed as defining a continuously changing subgoal (Bratko & Šuc, 2002; Šuc & Bratko, 1999a) (see also the use of flow tubes in dynamic plan execution (Hofmann & Williams, 2006)).

3. Subgoals and generalized trajectories are not sufficient to define a controller. A model of the systems dynamics is also required. Therefore, in addition to inducing subgoals or a generalized trajectory, this
   approach also requires learning approximate system dynamics, that is a model of the controlled system. 
   - Bratko and Šuc (2003) and Šuc and Bratko (1999b) use a combination of the Goldhorn (Križman & Džeroski, 1995) discovery program and locally weighted regression to build the model of the system’s dynamics. 
   - The next action is then computed “indirectly” by
     - computing the desired next state (e.g., next subgoal) 
     - and determining an action that brings the system to the desired next state. 
   - Bratko and Šuc also investigated building qualitative control strategies from operator traces (Bratko & Šuc, 2002).

---

An analog to this approach is **inverse reinforcement learning** (Abbeel & Ng, 2004; Atkeson & Schaal, 1997; Ng & Russell, 2000) where the reward function is learned. 

- Here, the learning the reward function corresponds to learning the human operator’s goals.

---

Isaac and Sammut (2003) uses an approach that is similar in spirit to Šuc and Bratko but incorporates classical control theory. Learned skills are represented by a two-level hierarchical decomposition with an **anticipatory goal level** and a **reactive control level**.

- The goal level models how the operator chooses goal settings for the control strategy and the control level models the operator’s reaction to any error between the goal setting and actual state of the system. 
  - For example, in flying, the pilot can achieve goal values for the desired heading, altitude, and airspeed by choosing appropriate values of turn rate, climb rate, and acceleration. 
  - The controls can be set to correct errors between the current state and the desired state of these goal-directing quantities. 
  - Goal models map system states to a goal setting. Control actions are based on the error between the output of each of the goal models and the current system state.
- The control level is modeled as a set of proportional integral derivative (PID) controllers, one for each
  control variable. A PID controller determines a control value as a linear function proportional to the error on a goal variable, the integral of the error, and the derivative of the error.

---

- Goal setting and control models are learned separately. The process begins be deciding which variables
  are to be used for the goal settings. For example, trainee pilots will learn to execute a “constant-rate turn,” that is, their goal is to maintain a given turn rate. A separate goal rule is constructed for each goal variable using a **model tree** learner (Potts & Sammut, 2005).
- A goal rule gives the setting for a goal variable and therefore, we can find the difference (error) between the current state value and the goal setting. The integral and derivative of the error can also be calculated. For example, if the set turn rate is $180^\circ$ min, then the error on the turn rate is calculated as the actual turn rate minus 180. The integral is then the running sum of the error multiplied by the time interval between time samples, starting from the first time sample of the behavioral trace, and the derivative is calculated as the difference between the error and previous error all divided by the time interval.
- For each control available to the operator, a model tree learner is used to predict the appropriate control setting. **Linear regression** is used in the leaf nodes of the model tree to produce linear equations whose coefficients are the P, I, and D of values of the PID controller. Thus the learner produces a collection of PID controllers that are selected according to the conditions in the internal nodes of the tree. In control theory, this is known as *piecewise linear control.*

---

Another indirect method is to learn a model of the dynamics of the system and use this to learn, in simulation, a controller for the system (Bagnell & Schneider, 2001; Ng, Jin Kim, Jordan, & Sastry, 2003).

- This approach does not seek to directly model the behavior of a human operator. A behavioral trace may be used to generate data for modeling the system but then a reinforcement learning algorithm is used to generate a policy for controlling the simulated system. The learned policy can then be transferred to the physical system. **Locally weighted regression** is typically used for system modeling, although model trees can also be used.



## 1.2 Behavioral Cloning Project

In this project,  we use deep neural networks and **convolutional neural networks** to **clone driving behavior.** We train, validate and test a model for **outputting a steering angle** to an autonomous vehicle.  

**Simulation:**

- We use image data and steering angles which are collected in the [Udacity simulator](https://github.com/udacity/self-driving-car-sim) to train a neural network and then use this model to drive the car autonomously **around the track in simulator.**

**Field Experiment:**

- A Pioneer robot equipped with ZED camera and NVIDIA TX2 controller.



## 1.3 Related Projects

- Open Source Self-Driving Car Project

#### 1.3.1 Open Source Self-Driving Car Project

***Project address:*** https://github.com/udacity/self-driving-car 

This project is maintained by Udacity and the aim of this project is to create a complete autonomous self-driving car **using deep learning and using ROS as middleware for communication.** 

---

**1. Sensors and components used in the Udacity self-driving car:**

- 2016 Lincoln MKZ : 

  This is the car that is going to be made autonomous. In other projects, we have saw the ROS interfacing of this car. We are using that project here too.

- Two Velodyne VLP-16 LiDARs

- Delphi radar

- Point Grey Blackfly cameras

- Xsens IMU

- Engine control unit ( ECU )

---

**2. dbw_mkz_ros package:**

This project uses the dbw_mkz_ros package to communicate from ROS to the Lincoln MKZ. In the previous section, we set up and worked with the dbw_mkz_ros package. 

- Here is the link to **obtain a dataset for training the steering model**: https://github.com/udacity/self-driving-car/tree/master/datasets . You will get a ROS launch file from this link to play with these bag files too.
- Here is the link to **get an already trained model** that can only be used for research purposes: https://github.com/udacity/self-driving-car/tree/master/steering-models . There is a ROS node for sending steering commands **from the trained model to the Lincoln MKZ.** Here, dbw_mkz_ros packages act as an intermediate layer between the trained model commands and the actual car.
- reference to implement the driving model using deep learning and the entire explanation for it are at https://github.com/thomasantony/sdc-live-trainer . 

---

**3. [Udacity Simulator](https://github.com/udacity/self-driving-car-sim)**

This simulator was built for [Udacity's Self-Driving Car Nanodegree](https://udacity.com/drive), to teach students how to train cars how to navigate road courses using deep learning. See more [project details here](https://github.com/udacity/CarND-Behavioral-Cloning-P3). All the assets in this repository require Unity. Please follow the instructions below for the full setup.

**Available Game Builds (Precompiled builds of the simulator)**

- Term 1
  - Instructions: Download the zip file, extract it and run the executable file.
  - Version 2, 2/07/17 [Linux](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip) [Mac](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-mac.zip) [Windows](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip)
  - Version 1, 12/09/16 [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip) [Mac](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip) [Windows 32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip) [Windows 64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

- Term 2
  - Please see the [Releases](https://github.com/udacity/self-driving-car-sim/releases) page for the latest version of the Term 2 simulator (v1.45, 6/14/17).
  - Source code can be obtained therein or also on the [term2_collection branch](https://github.com/udacity/self-driving-car-sim/tree/term2_collection).

- Term 3
  - Please see the [Releases](https://github.com/udacity/self-driving-car-sim/releases) page for the latest version of the Term 3 simulator (v1.2, 7/11/17).
  - Source code can be obtained therein or also on the [term3_collection branch](https://github.com/udacity/self-driving-car-sim/tree/term3_collection).

- System Integration / Capstone
  - Please see the [CarND-Capstone Releases](https://github.com/udacity/CarND-Capstone/releases) page for the latest version of the Capstone simulator (v1.3, 12/7/17). Source code can be obtained therein.

---

**4. Unity Simulator User Instructions (for advanced development)**

1. Clone the repository to your local directory, please make sure to use [Git LFS](https://git-lfs.github.com/) to properly pull over large texture and model assets.
2. Install the free game making engine [Unity](https://unity3d.com/), if you dont already have it. Unity is necessary to load all the assets.
3. Load Unity, Pick load exiting project and choice the `self-driving-car-sim` folder.
4. Load up scenes by going to Project tab in the bottom left, and navigating to the folder Assets/1_SelfDrivingCar/Scenes. To load up one of the scenes, for example the Lake Track, double click the file LakeTrackTraining.unity. Once the scene is loaded up you can fly around it in the scene viewing window by holding mouse right click to turn, and mouse scroll to zoom.

---

5. Play a scene. Jump into game mode anytime by simply clicking the top play button arrow right above the viewing window.
6. View Scripts. Scripts are what make all the different mechanics of the simulator work and they are located in two different directories, the first is Assets/1_SelfDrivingCar/Scripts which mostly relate to the UI and socket connections. The second directory for scripts is Assets/Standard Assets/Vehicle/Car/Scripts and they control all the different interactions with the car.
7. Building a new track. You can easily build a new track by using the prebuilt road prefabs located in Assets/RoadKit/Prefabs click and drag the road prefab pieces onto the editor, you can snap road pieces together easily by using vertex snapping by holding down "v" and dragging a road piece close to another piece.

---

**5. Related Resources:**

- https://github.com/udacity/CarND-Behavioral-Cloning-P3

- https://github.com/csharpseattle/CarND-Behavioral-Cloning

- https://github.com/upul/Behavioral-Cloning

- https://zhuanlan.zhihu.com/p/33222613

- https://github.com/navoshta/behavioral-cloning

- https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project

- https://github.com/UjjwalSaxena/Behavior-Cloning-DataSet-Ujjwal

- https://github.com/harveenchadha?tab=repositories

- https://github.com/mcarilli?tab=repositories

- https://github.com/darienmt/CarND-Behavioral-Cloning-P3





# 2. Related Work

## 2.1 ALVINN: AN AUTONOMOUS LAND VEHICLE IN A NEURAL NETWORK

***Dean A. Pomerleau. Technical report, Carnegie Mellon University, 1989.***

- In many ways, DAVE-2 was inspired by the pioneering work of Pomerleau [1] who in 1989 built the Autonomous Land Vehicle in a Neural Network (ALVINN) system. 
- It demonstrated that an **end-to-end trained neural network** can indeed steer a car on public roads.  ALVINN used a fully-connected network which is tiny by today’s standard.

<img src=imgs/p0_3.png height=300px/>

### 2.1.1 Introduction

- ALVINN (Autonomous Land Vehicle In a Neural Network) is a 3-layer back-propagation network designed for the task of road following. 
- Currently ALVINN takes images from a camera and a laser range finder as input and produces as output the direction the vehicle should travel in order to follow the road. 
- Training has been conducted using simulated road images. 
- Successful tests on the Carnegie Mellon autonomous navigation test vehicle indicate that **the network can effectively follow real roads under certain field conditions**. 
- The representation developed to perform the task differs dramatically when the network is trained under various conditions, suggesting the possibility of a novel adaptive autonomous navigation system capable of tailoring its processing to the conditions at hand.
  - *Autonomous navigation systems based on traditional image processing and pattern recognition techniques often perform well under certain conditions but have problems with others. Part of the difficulty stems from the fact that the processing performed by these systems remains fixed across various driving situations.*

### 2.1.2 Network Architecture

|                                                              |                                          |
| ------------------------------------------------------------ | ---------------------------------------- |
| 1. The activation level of each unit in the video input retina is proportional to the intensity in the blue color band of the corresponding patch of the image. The blue band of the color image is used because it provides the highest contrast between the road and the non-road. <br><br> 2. The activation level of each unit in the range finder input retina is proportional to the proximity of the corresponding area in the image.<br> <br> 3. The road intensity feedback unit indicates **whether** the road is lighter or darker than the non-road **in the previous image.** | <img src=imgs/alvinn.png height=400px /> |

---

**The output layer consists of 46 units, divided into two groups:**

- The first set of 45 units is a linear representation of the turn curvature along which the vehicle should travel in order to head towards the road center. 
- The middle unit represents the "travel straight ahead" condition while units to the left and right of the center represent successively sharper left and right turns. 

- The network is trained with a desired output vector of all zeros except for a "hill" of activation centered on the unit representing the correct **turn curvature**, which is the curvature which would bring the vehicle to the road center 7 meters ahead of its current position. 

  - More specifically, the desired activation levels for the nine units centered around the correct turn curvature unit are:

    ```
    0.10, 0.32, 0.61, 0.89, 1.00, 0.89, 0.61, 0.32 0.10
    ```

  - During testing, the turn curvature dictated by the network is taken to be the curvature represented by the output unit with the highest activation level.

---

The final output unit is a road intensity feedback unit which indicates whether the road is lighter or darker than the non-road **in the current image**. 

- During testing, the activation of the output road intensity feedback unit is recirculated to the input layer in the style of Jordan [Jordan, 1988] to aid the network's processing by providing rudimentary information concerning the relative intensities of the road and the non-road in the previous image.
- Using this extra information concerning the relative brightness of the road and the non-road, the network is better able to determine the correct direction for the vehicle to travel.

### 2.1.3 Training and Performance

**Limitations:**

- Training on actual road images is logistically difficult, because in order to develop a general representation, the network must be presented with a large number of training exemples depicting roads under a wide variety of conditions. 

- Collection of such a data set would be difficult, and **changes in parameters such as camera orientation would require collecting an entirely new set of road images.**


---

**Data Generator:**

- To avoid these difficulties we have developed a simulated road generator which creates road images to be used as training exemplars for the network. 
- Figure 2 depicts the video images of one real and one artificial road. Although not shown in Figure 2, the road generator also creates corresponding simulated range finder images. At the relatively low resolution being used it is difficult to distinguish between real and simulated roads.
- In addition, the network must not solely be shown examples of accurate driving, but also how to recover (i.e. return to the road center) once a mistake has been made.

<img src=imgs/p0_2.png height=300px/>

---

**Training:**

- Training involves first creating a set of 1200 road snapshots depicting roads with a wide variety of retinal orientations and positions, under a variety of lighting conditions and with realistic noise levels.
- After 40 epochs of training on the 1200 simulated road snapshots, the network correctly dictates a turn curvature within two units of the correct answer approximately 90% of the time on novel simulated road images.
- Specifically, the network can accurately drive the NAVLAB at a speed of 1/2 meter per second along a 400 meter path through a wooded area of the CMU campus under sunny fall conditions.

---

**Road intensity unit:**

- During the early stages of training, the input road intensity unit is given a random activation level. This is done to prevent the network from merely learning to copy the activation level of the input road intensity unit to the output road intensity unit, since their activation levels should almost always be identical because the relative intensity of the road and the non-road does not often change between two successive images. 
- Once the network has developed a representation that uses image characteristics to determine the activation level for the output road intensity unit, the network is given as input whether the road would have been darker or lighter than the non-road in the previous image. 
- Using this extra information concerning the relative brightness of the road and the non-road, the network is better able to determine the correct direction for the vehicle to travel.



### 2.1.4 Network Representation

The representation developed by the network to perform the road following task depends dramatically on the characteristics of the training set.

- When trained on examples of roads with a fixed width:

<img src=imgs/p0_4.png width=500px/>

---

- The network develops a very different representation when trained on snapshots with widely varying road widths.

<img src=imgs/p0_5.png width=500px/>



 

### 2.1.5 Discussion and Extensions

**1. Why not use real images？**（试图建立规则，并令机器学习规则：好or不好?）

- There are difficulties involved with training "on-the-fly" (不经过某种额外步骤而直接进行某项活动) with real images. 
- If the network is not presented with sufficient variability in its training exemplars to cover the conditions it is likely to encounter when it takes over driving from the human operator, **it will not develop a sufficiently robust representation and will perform poorly.** 
- In addition, the network must not solely be shown examples of accurate driving, but also how to recover (i.e. return to the road center) once a mistake has been made. 
- Partial initial training on **a variety of simulated road images** should help eliminate these difficulties and facilitate better performance.

---

**2. Is it necessary to use laser range finder?**

- Range data is in some sense less important for the task of road following than is the video data. 
  - The range data contains information concerning the position of obstacles in the scene, but nothing explicit about the location of the road. 
  - As a result, the range data is given less significance in the representation, as is illustrated by the relatively small magnitude weights from the range finder retina in the weight diagrams. 

- Figures 4 and 5 illustrate that the range finder connections do correlate with the connections from the video camera, and do contribute to choosing the correct travel direction. 
  - Specifically, in both figures, obstacles located outside the area in which the hidden unit expects the road to be located increase the hidden unit's activation level while obstacles located within the expected road boundaries inhibit the hidden unit. 
- However the contributions from the range finger connections aren't necessary for reasonable performance. When ALVINN was tested with normal video input but an obstacle-free range finder image as constant input, there was no noticeable degradation in driving performance. 
- Obviously under off-road driving conditions obstacle avoidance would become much more important and hence one would expect the range finder retina to play a much more significant role in the network's representation.

---

**Exploring different network architectures:**

1. Giving the network additional feedback information by using Elman's [Elman, 1988] technique of recirculating hidden activation levels.
2. Adding a second hidden layer to facilitate better internal representations.
3. Adding local connectivity to give the network a priori knowledge of the two dimensional nature of the input.

---> Deep CNN!

---

**Future work:**

- Road forks and intersections. 
  - Currently upon reaching a fork, the network may output two widely discrepant travel directions, one for each choice. The result is often an oscillation in the dictated travel direction and hence inaccurate road following. 
- Integrating a map into the system to enable global point-to-point path planning.

---



## 2.2 End to End Learning for Self-Driving Cars

*Mariusz Bojarski (NVIDIA Corporation), arXiv, 2016.*

PilotNet

### 2.2.1 Introduction

**Motivation:**

- The primary motivation for this work is to avoid the need to recognize specific human-designated features, such as lane markings, guard rails, or other cars, 
- and to avoid having to create a collection of “if, then, else” rules, based on observation of these features.

<img src=imgs/nvidia.png />

---

We trained a convolutional neural network (CNN) to **map raw pixels from a single front-facing camera directly to steering commands**.  The system operates at 30 frames per second (FPS).

- Compared to explicit decomposition of the problem, such as lane marking detection, path planning, and control, our **end-to-end** system **optimizes all processing steps simultaneously**. 
- We argue that this will eventually lead to better performance and smaller systems. 
  - Better performance will result because the internal components self-optimize to maximize overall system performance, instead of optimizing human-selected intermediate criteria, e. g., lane detection. *Such criteria understandably are selected for ease of human interpretation which doesn’t automatically guarantee maximum system performance.* 
  - Smaller networks are possible because the system learns to solve the problem with the minimal number of processing steps. 



### 2.2.2 Overview of the DAVE-2 System

**Data collection system:**

.center[<img src=imgs/p1_1.png width=400px />]

1. In order to make our system independent of the car geometry, we represent the steering command as 1/r where r is the turning radius in meters. 1/r smoothly transitions through zero from left turns (negative values) to right turns (positive values).
2. Training data contains single images sampled from the video, paired with the corresponding steering command (1/r). 

---

**Training system**

.center[<img src=imgs/p1_2.png height=300px />]

1. Training with data from only the human driver is not sufficient. **The network must learn how to recover from mistakes.** Otherwise the car will slowly drift off the road. 
   - **The training data is therefore augmented with additional images that show the car in different shifts from the center of the lane and rotations from the direction of the road.**
2. Additional shifts between the cameras and all rotations are simulated by viewpoint transformation of the image from the nearest camera. Precise viewpoint transformation requires 3D scene knowledge which we don’t have. We therefore approximate the transformation by assuming all points below the horizon are on flat ground and all points above the horizon are infinitely far away. This works fine for flat terrain but it introduces distortions for objects that stick above the ground, such as cars, poles, trees, and buildings. Fortunately these distortions don’t pose a big problem for network training. The steering label for transformed images is adjusted to one that would steer the vehicle back to the desired location and orientation in two seconds.



### 2.2.3 Data Collection

Training data was collected by driving on a wide variety of roads and in a diverse set of lighting and weather conditions. 

- Most road data was collected in central New Jersey, although highway data was also collected from Illinois, Michigan, Pennsylvania, and New York. 
- Other road types include 
  - two-lane roads (with and without lane markings), 
  - residential roads with parked cars, tunnels, and unpaved roads. 
- Data was collected in clear, cloudy, foggy, snowy, and rainy weather, both day and night. In some instances, the sun was low in the sky, resulting in glare reflecting from the road surface and scattering from the windshield.
- Drivers were encouraged to maintain full attentiveness, but otherwise drive as they usually do. As of March 28, 2016, about 72 hours of driving data was collected. (72x3600x10=2592000)

### 2.2.4 Network Architecture

.center[<img src=imgs/nvidia_network.png height=550px />]

---

class: middle

1. The input image is split into **YUV planes** and passed to the network.
2. The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture and to be accelerated via GPU processing.
3. The convolutional layers were designed to perform feature extraction and **were chosen empirically through a series of experiments that varied layer configurations.** We use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution (1 stride) with a 3×3 kernel size in the last two convolutional layers. 
4. The fully connected layers are designed to function as a controller for steering, but we note that by training the system end-to-end, **it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller.**
5. We train the weights of our network to minimize the mean squared error.



### 2.2.5 Training Details

**Data Selection：**

- Our collected data is labeled with road type, weather condition, and the driver’s activity (staying in a lane, switching lanes, turning, and so forth). 
- To train a CNN to do lane following we only select data where the driver was staying in a lane and discard the rest. 
- We then **sample that video at 10 FPS**. A higher sampling rate would result in including images that are highly similar and thus not provide much useful information.
- To remove a bias towards driving straight the **training data includes a higher proportion of frames that represent road curves.**

---

**Data Augmentation:**

- adding artificial shifts and rotations to teach the network how to recover from a poor position or orientation. **The magnitude of these perturbations is chosen randomly from a normal distribution.** 
- **The distribution has zero mean, and the standard deviation is twice the standard deviation that we measured with human drivers.** 
- Artificially augmenting the data does add undesirable artifacts as the magnitude increases (see Section 2).

### 2.2.6 Simulation

<img src=imgs/p1_5.png />

### 2.2.7 Evaluation

We estimate what percentage of the time the network could drive the car (autonomy). 

- The metric is determined by counting simulated human interventions (see Section 6). These interventions occur when the simulated vehicle departs from the center line by more than one meter.

- We assume that in real life an actual intervention would require a total of six seconds: this is the time required for a human to retake control of the vehicle, re-center it, and then restart the self-steering mode.

  .center[<img src=imgs/p1_metric.png />]

---

**Visualization of Internal CNN State:**

Figures 7 and 8 show the activations of the first two feature map layers for two different example inputs, an unpaved road and a forest.

<img src=imgs/p1_7.png width=350px /> <img src=imgs/p1_8.png width=350px />

This demonstrates that the CNN learned to detect useful road features on its own, i. e., with only the human steering angle as training signal. We never explicitly trained it to detect the outlines of roads, for example.

### 2.2.8 Conclusions

- We have empirically demonstrated that CNNs are able to learn the entire task of lane and road following without manual decomposition into 
  - road or lane marking detection, 
  - semantic abstraction, 
  - path planning, 
  - and control. 

- A small amount of training data from **less than a hundred hours** of driving was sufficient to train the car to operate in diverse conditions, on highways, local and residential roads in sunny, cloudy, and rainy conditions. 
- The CNN is able to learn meaningful road features from a very sparse training signal (steering alone).
  - The system learns for example to detect the outline of a road without the need of explicit labels during training. 
- More work is needed to improve the robustness of the network, to find methods to verify the robustness, and to improve visualization of the network-internal processing steps.



## 2.3 VisualBackProp: efficient visualization of CNNs for autonomous driving

*Mariusz Bojarski 2018 ICRA*

- We justify our approach with theoretical arguments and confirm that the proposed method **identifies sets of input pixels**, rather than individual pixels, **that collaboratively contribute to the prediction**. 
- It compares favorably to the layer-wise relevance propagation approach, i.e. it obtains similar visualization results and **achieves order of magnitude speed-ups**.

### 2.3.1 Introduction

**Motivation:**

One of the fundamental question that arises when considering CNNs as well as other deep learning models is:  **what made the trained neural network model arrive at a particular response?**

- This question is of particular **importance to the end-to-end systems**. The **interpretability** of such systems is limited. 
- Visualization tools aim at addressing this question by identifying parts of the input image that had the **highest influence** on forming the final prediction by the network. 
- It is also straightforward to think about visualization methods as **a debugging tool** that helps to understand if the network detects “reasonable” cues from the image to arrive at a particular decision. 

---

**Principle**

- The method relies on the intuition that when moving deeper into the network, **the feature maps contain less and less information which are irrelevant to the output**. 

  - Thus, the feature maps of the last convolutional layer should contain the most relevant information to determine the output. 

- At the same time, feature maps of **deeper layers have lower resolution**. 

  - The underlying idea of the approach is to **combine feature maps** containing only relevant information (deep ones) with the ones with higher resolution (shallow ones). 

- In order to do so, starting from the feature maps of the last convolutional layer, we “**backpropagate**” the information about the regions of relevance while simultaneously increasing the resolution, 

  - where the backpropagation procedure is not **gradient-based** (as is the case for example in sensitivity-based approaches [7, 8, 9]), but instead is **value-based**. We call this approach VisualBackProp.

  <img src=imgs/p2_1.png height=200px/>

---

**Advantages:**

- **real time:** Our visualization tool runs in real time and requires less computations than forward propagation. We empirically demonstrate that it is order of magnitude faster than the state-of-the-art visualization method, layer- wise relevance propagation (LRP) [10], while at the same time it leads to very similar visualization results. 
- **theoretical guarantees:** In the theoretical part of this paper we demonstrate that our algorithm finds for each pixel of the input image the approximated value of its contribution to the activations in the last convolutional layer of the network. To the best of our knowledge, the majority of the existing visualization techniques for deep learning, which we discuss in the Related Work section, lack theoretical guarantees, which instead we provide for our approach. 
- **general:** The visualization method for CNNs proposed in this paper was originally developed for CNN-based systems for steering autonomous cars, though it is highly general and can be used in other applications as well.



### 2.3.2 Related Work

- A notable approach [10] layer- wise relevance propagation, where the prediction is back- propagated without using gradients such that the relevance of each neuron is redistributed to its predecessors through a particular message-passing scheme relying on the conservation principle. 
  - The stability of the method and the sensitivity to different settings of the conservation parameters was studied in the context of several deep learning models [11]. 
  - The LRP technique was extended to Fisher Vector classifiers [12] and also used to explain predictions of CNNs in NLP applications [13]. 
  - An extensive comparison of LRP with other techniques, like the deconvolution method [14] and the sensitivity-based approach [8], which we also discuss next in this section, using an evaluation based on region perturbation can be found in [15]. This study reveals that **LRP provides better explanation of the CNN classification decisions than considered competitors.**

---

- Another approach [14] for understanding CNNs with max-pooling and rectified linear units (ReLUs) through visualization uses deconvolutional neural network [16] attached to the convolutional network of interest. 
  - This approach maps the feature activity in intermediate layers of a previously trained CNN back to the input pixel space using deconvolutional network, which performs successively repeated operations of i) unpooling, ii) rectification, and iii) filtering. 
  - Since this method identifies structures within each patch that stimulate a particular feature map, it differs from previous approaches [17] which instead identify patches within a data set that stimulate strong activations at higher layers in the model. 
  - The method can also be interpreted as providing an approximation to partial derivatives with respect to pixels in the input image [8]. 
  - One of the shortcomings of the method is that it enables the visualization of only a single activation in a layer (all other activations are set to zero). 

---

- There also exist other techniques for inverting a modern large convolutional network with another network, e.g. a method based on up- convolutional architecture [18], where as opposed to the previously described deconvolutional neural network, the **up- convolutional network is trained**. This method inverts deep image representations and obtains reconstructions of an input image from each layer.
  - The fundamental difference between the LRP approach and the deconvolution method lies in how the responses are projected towards the inputs. The latter approach solves the optimization problems to reconstruct the image input while the former one aims to reconstruct the classifier decision (the details are well-explained in [10]).
- Guided backpropagation [19] extends the deconvolution approach by combining it with a simple technique visualizing the part of the image that most activates a given neuron using a backward pass of the activation of a single neuron after a forward pass through the network. Finally, the recently published method [20] based on the prediction difference analysis [21] is a probabilistic approach that extends the idea in [14] of visualizing the probability of the correct class using the occlusion of the parts of the image. The approach highlights the regions of the input image of a CNN which provide evidence for or against a certain class.

---

- Understanding CNNs can also be done by visualizing output units as distributions in the input space via output unit sampling [22]. 
  - However, computing relevant statistics of the obtained distribution is often difficult. This technique cannot be applied to deep architectures based on auto-encoders as opposed to the subsequent work [23, 24], where the authors visualize what is activated by the unit in an arbitrary layer of a CNN in the input space (of images) via an activation maximization procedure that looks for input patterns of a bounded norm that maximize the activation of a given hidden unit using gradient ascent. 
  - This method extends previous approaches [25]. The gradient-based visualization method [24] can also be viewed as a generalization of the deconvolutional network reconstruction procedure [14]as shown in subsequent work [8]. The requirement of careful initialization limits the method [14]. The approach was applied to Stacked Denoising Auto-Encoders, Deep Belief Networks and later on to CNNs [8]. 
  - Finally, sensitivity-based methods [8, 7, 9]) aim to understand how the classifier works in different parts of the input domain by computing scores based on partial derivatives at the given sample.

---

- Some more recent gradient-based visualization techniques for CNN-based models not mentioned before include Grad- CAM [26], which is an extension of the Class Activation Mapping (CAM) method [27]. The approach heavily relies on the construction of weighted sum of the feature maps, where the weights are global-average-pooled gradients obtained through back-propagation. The approach lacks the ability to show fine-grained importance like pixel-space gradient visualization methods [19, 14] and thus in practice has to be fused with these techniques to create high-resolution class- discriminative visualizations.
- Finally, other approaches for analyzing neural networks include quantifying variable importance in neural networks [28, 29], extracting the rules learned by the decision tree model that is fitted to the function learned by the neural network [30], applying kernel analysis to understand the layer-wise evolution of the representation in a deep network [31], analyzing the visual information in deep image representations by looking at the inverse representations [32], applying contribution propagation technique to provide per-instance explanations of predictions [33] (the method relies on the technique of [34]), or visualizing particular neurons or neuron layers [2, 35]. Finally, there also exist more generic tools for explaining individual classification decisions of any classification method for single data instances, like for example [7].



### 2.3.3 Visualization Method

Our method combines feature maps from deep convolutional layers that contain mostly relevant information, but are low-resolution, with the feature maps of the shallow layers that have higher resolution but also contain more irrelevant information. (*The last layer has prediction information and the previous layer has position. back-activation*)

- This is done by “back- propagating” the information about the regions of relevance while simultaneously increasing the resolution. 

<img src=imgs/p2_2_1.png height=350px/>

---

- The method utilizes the forward propagation pass, which is already done to obtain a prediction, i.e. we do not add extra forward passes. 
- The method then uses the feature maps obtained **after each ReLU layer** (thus these feature maps are already thresholded).   
- In the first step, the **feature maps** from each layer are averaged, resulting in **a single feature map per layer.** 

- Next, the averaged feature map of the deepest convolutional layer is scaled up to the size of the feature map of the previous layer. 
  - This is done using **deconvolution** with filter size and stride that are the same as the ones used in the deepest convolutional layer (for deconvolution we always use the same filter size and stride as in the convolutional layer which outputs the feature map that we are scaling up with the deconvolution). 
  - In deconvolution, all weights are set to 1 and biases to 0. 
- The obtained scaled- up averaged feature map is then **point-wise multiplied** by the averaged feature map from the previous layer. The resulting image is again scaled via deconvolution and multiplied by the averaged feature map of the previous layer exactly as described above. 

- In the end, we obtain a mask of the size of the input image, which we **normalize to the range [0, 1]**.

---

- On the left side the figure shows the averaged feature maps of all the convolutional layers from the input (top) to the output (bottom). 
- On the right side it shows the corresponding intermediate masks. Thus on the right side we show step by step how the mask is being created when moving from the network’s output to the input. 
- Comparing the two top images clearly reveals that **many details were removed** in order to obtain the final mask.

.center[<img src=imgs/p2_2_2.png height=400px/>]

---

**The implementation of deconvolution:**

- https://www.jianshu.com/p/f0674e48894c 



### 2.3.4 Theoretical Analysis 

We now present the theoretical guarantee for the algorithm (all proofs are skipped and are available upon request). We emphasize that our theoretical analysis does not rely on computing the sensitivity of any particular cost function with respect to the changes of values of particular input neurons. So we will not focus on computing the gradients. The reason for that is that even if the gradients are large, the actual contribution of the neuron might be small. Instead our proposed method is measuring the actual contribution that takes into account the “collaborative properties” of particular neurons. This is measured by the ability of particular neurons to substantially participate in these weighted inputs to neurons in consecutive layers that *themselves have higher impact on the form of the ultimate feature maps than others.*

Consider a convolutional neural network N with ReLU nonlinear mappings. We assume that no pooling mechanism is used and the strides are equal to one (the entire analysis can be repeated for arbitrary stride values). Furthermore:

<img src=imgs/p2_3.png />

- $\gamma(X)$ is the value of the input pixel X;
- $w_e$ stands for the weight of an edge $e$;
- let $e$ be an edge from some neuron $v'$ to $v$. Then $\gamma_e $ will denote the weighted input to neuron $v$ along edge $e$, $a_e$ will denate the activation of $v$, and $b_e$ will denote the bias of $v$.

***Lemma 1:*** Consider the general neural network architecture described above. Then the contribution of the input pixel X to the last layer of feature maps is given as:
$$
\phi^{N}(X) = \gamma(X) \sum_{P\in\textbf{P}}\prod_{e\in P}\frac{a_e}{a_e+b_e} w_e
$$
where $\gamma(X)$ is the value of pixel $X$ and $\textbf{P}$ is a family of paths from $X$ to the last layer of the network.

***Theorem 1***: For a fixed CNN N considered in this paper there exits a universal constant c >0 such that the values of the input neurons conputed by the VisualBackProp algorithm are of the form:
$$
\phi^{N}_{V BP}(X) = c \gamma(X) \sum_{P\in\textbf{P}}\prod_{e\in P}a_e
$$
The statement above shows that the values computed for pixels by the VisualBackProp algorithm are related to the flow contribution from that pixels in the corresponding graphical model and thus, according to our analysis, measure their importance. The formula on $\phi^{N}_{V BP}(X)$ is similar to the one on $\phi^{N}(X)$, but gives rise to a much more efficient algorithm and leads to tractable theoretical analysis. Note that the latter one can be obtained from the former one by multiplying each term of the inner products by $\frac{w_e}{a_e+b_e}$ and then rescaling by a multiplicative factor of $\frac{1}{c}$. Rescaling does not have any impact on quality since it is conducted in exactly the same way for all the input neurons. Finally, the following observation holds. 

***Remark 1:*** Note that for small kernels the number of paths considered in the formula on $\phi^{N}(X)$ is small (since the degrees of the corresponding multipartite graph are small) thus in practice the difference between formula on $\phi^{N}_{V BP}(X)$ and the formula on φN(X) coming from the re-weighting factor $\frac{w_e}{a_e+b_e}$ is also small. Therefore for small kernels the VisualBackProp algorithm computes very good approximations of input neurons’ contributions to the activations in the last layer. 



### 2.3.5 Experiments

- The codes of VisualBackProp are already publicly released at https://github.com/mbojarski/VisualBackProp. 

- The first set of experiments is conducted on the PilotNet and aims at validating whether VisualBackProp is able to **capture the parts of the image that are indeed relevant for steering the self-driving car.** 
- The next set of experiments were performed on the **Udacity self-driving car data set** (Udacity Self Driving Car Dataset 3-1: El Camino). 
  - We qualitatively compare our method with LRP, the state-of-the-art deep learning visualization technique, (we use implementation as given in Equation 6 from [15]; similarly to the authors, we use $ \epsilon= 100$) and we also compare their running times. 
- We finally show experimental results on the **task of the classification** of traffic signs on the German Traffic Sign Detection Benchmark data set (http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) and also ImageNet data set (http://image-net.org/challenges/LSVRC/2016/).
- Therefore we demonstrate the applicability of the technique to a **wide-range of learning problems**.

#### A. PilotNet: a system for autonomous driving

.center[<img src=imgs/p2_5.png height=500px/>]

---

We segmented the input image that is presented to PilotNet into **two classes**.

- Class 1 is meant to include all the regions that have a significant effect on the steering angle output by PilotNet. These regions include all the pixels that correspond to locations **where the visualization mask is above a threshold.**(Fig.7c) (*These regions are then dilated by 30 pixels to counteract the increasing span of the higher-level feature map layers with respect to the input image. The exact amount of dilation was determined empirically.*) 
- The second class includes all pixels in **the original image minus the pixels in Class 1**. 

If the objects found by our method indeed dominate control of the output steering angle, we would expect the following: 

- if we create an image in which we uniformly **translate only the pixels in Class 1** while maintaining the position of the pixels in Class 2 and use this new image as input to PilotNet, we would **expect a significant change** in the steering angle output. 
- However, if we instead **translate the pixels in Class 2** while keeping those in Class 1 fixed and feed this image into PilotNet, then we would expect minimal change in PilotNet’s output.

---

Figure 7 illustrates the process described above:

- The top image is an original input image. 
- The next image shows highlighted regions that were identified using VisualBackProp. 
- The next image illustrates Class 1. 
- The bottom image shows the input image with a relevant region from the third image shifted.

<img src=imgs/p2_7.png height=400px/>

---

Figure 8 shows plots of PilotNet steering output as a function of pixel shift in the input image:

- Shifting the pixels belonging to Class 1 results in a linear change in **steering angle that is nearly as large as that which occurs when we shift the entire image.** 

- Shifting just the background pixels has **a much smaller effect** on the steering angle. 

- We are thus confident that our method **does indeed find the most important regions** in the image for determining steering.

  <img src=imgs/p2_8.png />



#### B. Udacity self-driving car data set

- NetHVF input image has approximately two times higher vertical field of view, but then is scaled down by that factor. 
- The networks are trained with stochastic gradient descent (SGD) and the mean squared error (MSE) cost function for 32 epochs.

.center[<img src=imgs/p2_t1.png height=350px/>]

- Each layer except for the last fully-connected layer is followed by a RELU.
- Each convolution layer is preceded by a batch normalization layer.
- layer output size is nxhxw
- cropping and scaling the lower part of the images to a 640×135 size for network NetSVF and 351×135 size for network NetHVF.

---

Figures 10 illustrates that the CNN learned to recognize lane markings, the most relevant visual cues for steering a car. 

- It also shows that **the field of view affects the visualization results significantly**. 

<img src=imgs/p2_10.png height=450px/>

---

Figure 11 shows two consecutive frames:

- On the second frame in Figure 11, the lane marking on the left side of the road disappears, which causes the CNN to change the visual cue it focuses on from the lane marking on the left to the one on the right. 

<img src=imgs/p2_11.png />

---

Figure 12 corresponds to sharp turns:

- The images in the top row of Figure 12 demonstrate the correlation between the high prediction error of the network and the low-quality visual cue it focuses on.

<img src=imgs/p2_12.png />

---

Finally, in Figure 13 we demonstrate that the CNN has learned to **ignore horizontal lane markings** as they are not relevant for steering a car, even though it was trained only with the images and the steering wheel angles as the training signal. 

<img src=imgs/p2_13.png />

---

Figure 14 similarly shows that the CNN learned to ignore the horizontal lines, however, as the visualization shows, it does not identify lane markings as the relevant visual cues but **other cars instead**.

<img src=imgs/p2_14.png />

---

**Time:**

- The average time of computing a mask for VisualBackProp was equal to **2.0 ms**. 
- It was **24.6 ms** for the LRP. 
- The VisualBackProp is therefore on average 12 times faster than LRP. 
- At the same time, as demonstrated in Figures 10–14, VisualBackProp generates visualization masks that are very similar to those obtained by LRP.



#### C. German Traffic Sign Detection benchmark data set and ImageNet data set

ARCHITECTURE OF THE NETWORK USED FOR SIGN CLASSIFICATION. 

<img src=imgs/p2_t2.png />

- EACH LAYER EXCEPT FOR THE LAST FULLY-CONNECTED LAYER IS FOLLOWED BY A RELU. 
- THE LAST FULLY-CONNECTED LAYER IS FOLLOWED BY A LOGSOFTMAX.
- EACH CONVOLUTION LAYER IS PRECEDED BY A BATCH NORMALIZATION LAYER.

---

- We next (Figure 15) demonstrate the performance of VisualBackProp and LRP on German Traffic Sign Detection Benchmark data set. 

<img src=imgs/p2_15.png height=450px/>

---

- Finally, we show the performance of VisualBackProp on ImageNet data set. The network here is a ResNet-200 [36].

<img src=imgs/p2_16.png height=500px/>





# 3. Implementation

## 3.1 Simulation Dataset

*For more details of the simulator used in this project,  see Section 1.3.1.* 

### 3.1.1 Data Collection

Data collection was done using the simulator in 'Training Mode' :

1. At first, I gathered 1438 images from a full lap around track. But I can not always keep the car driving at the center line of the road. This is not a good dataset and previous experience have shown that this kind of data will make the car pull too hard in testing. So I discard it. 
2. Then I collected a new data set which contains 2791(x3) images by driving the car travel the full track two times. (backed up in carnd folder)
   - The total number of images is about 8000 if two side images are counted.

tips:

- Use left and right images and their corresponding steering angle is the original steering angle adding (left) or subtracting (right) a correction angle.
- Collecting more data at the easily failed place such as the turn track.
- Collecting the reverse driving data.
- After much trial and error I found that it was not necessary to gather images from a full lap around each track. For the first track I drove just past the bridge to the first dirt patch and for the mountain track I drove only half-way around.  

### 3.1.2 Preprocessing Images

Images were preprocessed before feeding to the neural network. 

#### A. Choosing Color Space

There are three types of color space for image representation.

![](/home/ubuntu16/Behavioral_Cloning/imgs/color_space.png)

- Please keep in mind that **the colorspace of training image loaded by cv2 is BGR**.  However, when the trained network predicts the steering angles at the testing stage,  `drive.py` **loads image with RGB** colorspace by PIL. 
- Therefore, in the training stage, we will convert the BGR image loaded by OpenCV to RGB before feeding the real image into the training network to ensure the consistency of the colorspace between the training stage and testing satge.

To Do List:

- [ ] *I tried different color spaces and found that the model generalized best using images in the BGR color space. -- need to be verified.* 
- [ ] Training with the **YUV color space** gave erratic steering corrections resulting in too much side-to-side movement of the car.  (test)
- [ ] In the **RGB color space** some areas of track required many more training images to navigate correctly, particularly the areas with dirt patches on the side of the road.  (test)

 

#### B. Image Cropping

- I crop the unnecessary portions of the image (background of sky, trees, mountains, hood of car) **taking 50 pixels off the top of the image and 20 pixels off the bottom.**  

<img src=imgs/cropped.png width=600px/>

```python
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model = Sequential()
model.add(Cropping2D(cropping=((50, 20),(0,0)), input_shape=(160,320,3)))
# output the intermediate layer! 
## can be used to show learned features for any layer
layer_output = backend.function([model.layers[0].input], [model.layers[0].output])
# note that the shape of the image suited for the network!
cropped_image = layer_output([image[None,...]])[0][0]
# convert to uint8 for visualization
cropped_image = np.uint8(cropped_image)
```



#### C.  Image Resizing

testing...



### 3.1.3 Data Augmentation

There are four kinds of methods for data augmentation.  Images could be augmented by 

- flipping horizontally, 
- blurring, 
- changing the overall brightness, 
- or applying shadows to a portion of the image.  

***Remark:***

- When reading original data from CSV file,  flipped_flag, shadow_flag, bright_flag, blur_flag will be assigned to each image. The image processing will be done when generating practical data sets.

#### 1. Flipping horizontally

By adding a flipped image for every original image the data set size was effectively doubled. 

In the end, adding horizontally flipped images was the only augmentation that was necessary.  

***Remark:***

1. When reading original data from the CSV file, the path of the flipped image is the same as the original image. However, a flipped flag (equals to 1) will be assigned to the flipped image and the flag of the original image is 0. 
2. Meanwhile, we set the negative value of the corresponded original image label as the label of the flipped image.   
3. Finally, the corresponding flipping operation will be done for each image according to this flag. 

```python
image = cv2.flip(image, 1)
```

---

#### 2. Changing brightness

![](/home/ubuntu16/Behavioral_Cloning/imgs/brightness.png)



#### 3. Bluring

![](/home/ubuntu16/Behavioral_Cloning/imgs/blur.png)

```python
kernel_size = (np.random.randint(1,5)*2) +1 
blur = cv2.GaussianBlur(rgb, (kernel_size,kernel_size), 0)
```



#### 4. Random shadow

![](/home/ubuntu16/Behavioral_Cloning/imgs/shadow.png)



***Remark:***

- See the functions `random_blur()`, `random_brightness()`, and `random_shadow()` in the file `data.py` for augmentation code. 
- Visualization of data augmentation can be found in the Jupyter notebooks `Visualization.ipynb` .

???

####5. Data Distribution

One improvement that was found to be particularly effective was to fix the poor distribution of the data.  A disproportionate number of steering angles in the data set are at or near zero.  To correct this:

- steering angles are separated into 25 bins. 
- Bins with a count less than the mean are augmented 
- while bins with counts greater than the mean are randomly pruned to bring their counts down. 

Those operations equalize the distribution of steering angles across all bins.  

<img src= imgs/original_distribution.png width=240px/> <img src=imgs/augmented_distribution.png width=240px/> <img src=imgs/distribution_corrected.png width=240px/>



## 3.2 Network Architecture

- Modified ALVINN
- PilotNet
- Modified PilotNet

### 3.2.1 Fully Connected Network

**A fully connected neural network with one hidden layer (100 units)：** 

- After many attempts, a normalization layer was added to the network.  The mse loss decreased to 8.8145 and the mae loss decreased to 1.9536. It is worth noting that the mae loss almost no longer reduced after 10 training epochs. Finally, the car can drive at the straight track but the steering angle is still too large at the turn of the track. 

- The parameter file is saved as `fcnet-normalize.h5`

```bash
   Epoch 19/20
   26/26 [==============================] - 13s 500ms/step - loss: 8.8145 - mean_absolute_error: 1.8013 - val_loss: 18.5598 - val_mean_absolute_error: 1.9536
   Epoch 20/20
   26/26 [==============================] - 13s 482ms/step - loss: 8.5064 - mean_absolute_error: 1.7278 - val_loss: 21.0641 - val_mean_absolute_error: 2.0854
```

```python
python drive.py fcnet-normalize.h5
```

---

2. Then i trained again with the same setup, The result seems very similar to the last training. However, the initial predicted steering angle is too large to drive the car out of the track.

```bash
   Epoch 15/20
   27/27 [==============================] - 12s 450ms/step - loss: 5.1247 - mean_absolute_error: 1.3832 - val_loss: 15.7272 - val_mean_absolute_error: 1.6583
   Epoch 16/20
   27/27 [==============================] - 12s 454ms/step - loss: 3.6974 - mean_absolute_error: 1.2955 - val_loss: 20.2050 - val_mean_absolute_error: 1.7890
   Epoch 20/20
   27/27 [==============================] - 12s 437ms/step - loss: 4.2537 - mean_absolute_error: 1.1959 - val_loss: 13.9532 - val_mean_absolute_error: 1.5473
  
```

---

3. Based on the above setup, the hidden layer units are increased to 1000 from 100. But the network converge more slowly and the final results is not good at all. The predicted steering angles is about 7.

---

class: middle, center

**Video:**

<video width="560" height="420" controls>
    <source src="videos/fc_10.mp4" type="video/mp4">
</video>

### 3.2.2 PilotNet

1. The model and training process are listed below. The final trained model works very well. 
2. All the saved model (even the model saved after 1 epoch)  can keep the car drving along the center line of the track for the full lap without any fails. 
3. The desired velocity of the car can be set arbitrary from 0 to 30 mph.

```bash
Reading data from csv file...
Reading is done.
EPOCHS: 20
Training Set Size: 6698
Valization Set Size: 1675
Batch Size: 256
/home/ubuntu16/Behavioral_Cloning/data.py:102: RuntimeWarning: divide by zero encountered in true_divide
  copy_times = np.float32((desired_per_bin-hist)/hist)
Training set size now: 6122
Using TensorFlow backend.
```

---

```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              9834636   
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 10,088,055
Trainable params: 10,088,055
Non-trainable params: 0
_________________________________________________________________
```

---

**Training process:**

```bash
Training with 24 steps, 7 validation steps.
Epoch 1/20
24/24 [==============================] - 11s 438ms/step - loss: 0.2917 - mean_absolute_error: 0.4121 - val_loss: 0.0490 - val_mean_absolute_error: 0.1765
Epoch 2/20
24/24 [==============================] - 9s 395ms/step - loss: 0.0581 - mean_absolute_error: 0.1890 - val_loss: 0.0403 - val_mean_absolute_error: 0.1485
Epoch 3/20
24/24 [==============================] - 10s 402ms/step - loss: 0.0366 - mean_absolute_error: 0.1501 - val_loss: 0.0293 - val_mean_absolute_error: 0.1267
Epoch 4/20
24/24 [==============================] - 10s 404ms/step - loss: 0.0275 - mean_absolute_error: 0.1297 - val_loss: 0.0334 - val_mean_absolute_error: 0.1427
Epoch 5/20
24/24 [==============================] - 10s 401ms/step - loss: 0.0253 - mean_absolute_error: 0.1225 - val_loss: 0.0268 - val_mean_absolute_error: 0.1228
Epoch 6/20
24/24 [==============================] - 10s 406ms/step - loss: 0.0201 - mean_absolute_error: 0.1097 - val_loss: 0.0248 - val_mean_absolute_error: 0.1173
Epoch 7/20
24/24 [==============================] - 10s 405ms/step - loss: 0.0179 - mean_absolute_error: 0.1029 - val_loss: 0.0219 - val_mean_absolute_error: 0.1156
Epoch 8/20
24/24 [==============================] - 10s 406ms/step - loss: 0.0139 - mean_absolute_error: 0.0891 - val_loss: 0.0252 - val_mean_absolute_error: 0.1230
Epoch 9/20
24/24 [==============================] - 10s 407ms/step - loss: 0.0130 - mean_absolute_error: 0.0862 - val_loss: 0.0239 - val_mean_absolute_error: 0.1229
Epoch 10/20    (saved)
24/24 [==============================] - 10s 404ms/step - loss: 0.0114 - mean_absolute_error: 0.0803 - val_loss: 0.0201 - val_mean_absolute_error: 0.1115
Epoch 11/20
24/24 [==============================] - 10s 409ms/step - loss: 0.0104 - mean_absolute_error: 0.0764 - val_loss: 0.0229 - val_mean_absolute_error: 0.1213
Epoch 12/20
24/24 [==============================] - 10s 406ms/step - loss: 0.0097 - mean_absolute_error: 0.0743 - val_loss: 0.0276 - val_mean_absolute_error: 0.1307
Epoch 13/20
24/24 [==============================] - 10s 406ms/step - loss: 0.0093 - mean_absolute_error: 0.0729 - val_loss: 0.0240 - val_mean_absolute_error: 0.1216
Epoch 14/20
24/24 [==============================] - 10s 403ms/step - loss: 0.0084 - mean_absolute_error: 0.0676 - val_loss: 0.0219 - val_mean_absolute_error: 0.1159
Epoch 15/20
24/24 [==============================] - 10s 405ms/step - loss: 0.0067 - mean_absolute_error: 0.0606 - val_loss: 0.0206 - val_mean_absolute_error: 0.1129
Epoch 16/20
24/24 [==============================] - 10s 405ms/step - loss: 0.0066 - mean_absolute_error: 0.0616 - val_loss: 0.0284 - val_mean_absolute_error: 0.1347
Epoch 17/20
24/24 [==============================] - 10s 406ms/step - loss: 0.0059 - mean_absolute_error: 0.0581 - val_loss: 0.0203 - val_mean_absolute_error: 0.1136
Epoch 18/20
24/24 [==============================] - 10s 404ms/step - loss: 0.0053 - mean_absolute_error: 0.0531 - val_loss: 0.0206 - val_mean_absolute_error: 0.1141
Epoch 19/20
24/24 [==============================] - 10s 407ms/step - loss: 0.0049 - mean_absolute_error: 0.0522 - val_loss: 0.0223 - val_mean_absolute_error: 0.1164
Epoch 20/20
24/24 [==============================] - 10s 400ms/step - loss: 0.0043 - mean_absolute_error: 0.0484 - val_loss: 0.0203 - val_mean_absolute_error: 0.1128
```

---

<video width="560" height="420" controls>
    <source src="videos/PilotNet.mp4" type="video/mp4">
</video>

### 3.2.3 Modified PilotNet

Overall, The modified nvidia network is very similar to the original one.  

- In this version, all the strides of filters are set to be 1. The corresponding max pooling operation is added after the first three convolution layers to ensure the output is consistent with the original network.
- Dropout layers were used in between the fully connected layers to reduce overfitting.  

The final performance of the trained network is not as good as the original one.

---

**Model:**

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

---

**Training**

```bash
25/25 [==============================] - 13s 503ms/step - loss: 0.2863 - mean_absolute_error: 0.4376 - val_loss: 0.1224 - val_mean_absolute_error: 0.2843
Epoch 2/20
25/25 [==============================] - 10s 417ms/step - loss: 0.0728 - mean_absolute_error: 0.2170 - val_loss: 0.0419 - val_mean_absolute_error: 0.1532
Epoch 3/20
25/25 [==============================] - 11s 429ms/step - loss: 0.0437 - mean_absolute_error: 0.1646 - val_loss: 0.0278 - val_mean_absolute_error: 0.1239
Epoch 4/20
25/25 [==============================] - 10s 416ms/step - loss: 0.0368 - mean_absolute_error: 0.1506 - val_loss: 0.0297 - val_mean_absolute_error: 0.1264
Epoch 5/20
25/25 [==============================] - 10s 414ms/step - loss: 0.0327 - mean_absolute_error: 0.1429 - val_loss: 0.0325 - val_mean_absolute_error: 0.1338
Epoch 6/20
25/25 [==============================] - 10s 411ms/step - loss: 0.0302 - mean_absolute_error: 0.1376 - val_loss: 0.0253 - val_mean_absolute_error: 0.1191
Epoch 7/20
25/25 [==============================] - 11s 447ms/step - loss: 0.0284 - mean_absolute_error: 0.1322 - val_loss: 0.0267 - val_mean_absolute_error: 0.1253
Epoch 8/20
25/25 [==============================] - 10s 408ms/step - loss: 0.0257 - mean_absolute_error: 0.1258 - val_loss: 0.0301 - val_mean_absolute_error: 0.1353
Epoch 9/20
25/25 [==============================] - 11s 422ms/step - loss: 0.0242 - mean_absolute_error: 0.1223 - val_loss: 0.0301 - val_mean_absolute_error: 0.1328
Epoch 10/20
25/25 [==============================] - 10s 410ms/step - loss: 0.0231 - mean_absolute_error: 0.1192 - val_loss: 0.0230 - val_mean_absolute_error: 0.1184
Epoch 11/20
25/25 [==============================] - 11s 423ms/step - loss: 0.0221 - mean_absolute_error: 0.1167 - val_loss: 0.0280 - val_mean_absolute_error: 0.1290
Epoch 12/20
25/25 [==============================] - 10s 418ms/step - loss: 0.0225 - mean_absolute_error: 0.1171 - val_loss: 0.0274 - val_mean_absolute_error: 0.1286
Epoch 13/20
25/25 [==============================] - 10s 410ms/step - loss: 0.0215 - mean_absolute_error: 0.1139 - val_loss: 0.0269 - val_mean_absolute_error: 0.1265
Epoch 14/20
25/25 [==============================] - 10s 408ms/step - loss: 0.0198 - mean_absolute_error: 0.1097 - val_loss: 0.0238 - val_mean_absolute_error: 0.1191
Epoch 15/20
25/25 [==============================] - 10s 418ms/step - loss: 0.0190 - mean_absolute_error: 0.1087 - val_loss: 0.0284 - val_mean_absolute_error: 0.1319
Epoch 16/20
25/25 [==============================] - 10s 417ms/step - loss: 0.0178 - mean_absolute_error: 0.1041 - val_loss: 0.0277 - val_mean_absolute_error: 0.1280
Epoch 17/20
25/25 [==============================] - 10s 419ms/step - loss: 0.0181 - mean_absolute_error: 0.1052 - val_loss: 0.0230 - val_mean_absolute_error: 0.1175
Epoch 18/20
25/25 [==============================] - 10s 409ms/step - loss: 0.0165 - mean_absolute_error: 0.1006 - val_loss: 0.0228 - val_mean_absolute_error: 0.1164
Epoch 19/20
25/25 [==============================] - 10s 411ms/step - loss: 0.0157 - mean_absolute_error: 0.0977 - val_loss: 0.0238 - val_mean_absolute_error: 0.1204
Epoch 20/20
25/25 [==============================] - 10s 415ms/step - loss: 0.0152 - mean_absolute_error: 0.0967 - val_loss: 0.0250 - val_mean_absolute_error: 0.1219

```

---

**Remark:**

1. Two laps of track images are used. The distribution of the data was fixed duiring training. The data set did not flipped.

2. You must make the car driving along the center of the track. Otherwise, the car will drive out of the track will testing.

3. finall loss is 0.02 

4. Validation accuracy was not a good indicator of the performance of the network.  It was better to watch for overfitting by comparing the training mean squared error (MSE) with the validation MSE.  A climbing validation MSE while the training MSE was still decreasing was a sign that the model was overfitting.  Dropout was used to combat this but I also limited the number of epochs to get the best performing model.


------

<video width="560" height="420" controls>
    <source src="videos/modified_nvidia_model.mp4" type="video/mp4">
</video>

---

## 3.3 Model Visualization

### 3.3.1 Feature Representation

```python
model = Sequential()
model.add(Cropping2D(cropping=((50, 20),(0,0)), input_shape=(160,320,3)))

# output the intermediate layer! 
## can be used to show learned features for any layer
layer_output = backend.function([model.layers[0].input], [model.layers[0].output])

# note that the shape of the image suited for the network!
cropped_image = layer_output([image[None,...]])[0][0]

```



### Using Saved Model

h5 model

```python
import h5py
from keras.models import load_model

from PIL import Image
import numpy as np

# check that model Keras version is same as local Keras version
f = h5py.File(args.model, mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')

if model_version != keras_version:
    print('You are using Keras version ', keras_version,
          ', but the model was built using ', model_version)

# loading saved model (model_path)
model = load_model(args.model)

## Reciving image data and do prediction using saved model
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        
        # Prediction
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame (from the car's view)
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # do something
        pass
  
```



# 4. Advanced Work

## 4.1 DroNet: Learning to Fly by Driving

*Antonio Loquercio∗, Ana I. Maqueda †, Carlos R. del-Blanco †, and Davide Scaramuzza ICRA2018 -RAL*

- For supplementary video see:https://youtu.be/ow7aw9H4BcA.
- The project’s code, datasets and trained models are available at: http://rpg.ifi.uzh.ch/dronet.html

#### Abstract

**Background**:

- Civilian drones are soon expected to be used in a wide variety of tasks, such as aerial surveillance, delivery, or monitoring of existing architectures. Nevertheless, their deployment in urban environments has so far been limited. Indeed, in unstructured and highly dynamic scenarios, drones face numerous challenges to navigate autonomously in a feasible and safe way. 

**Methodology**:

- In contrast to traditional **“map-localize-plan”** methods, this paper explores a **data-driven approach** to cope with the above challenges. 
- To accomplish this, we propose DroNet: a convolutional neural network that can safely drive a drone through the streets of a city. 
- Designed as a fast 8-layers residual network, DroNet produces two outputs for each single input image: a steering angle to keep the drone navigating while avoiding obstacles, and **a collision probability** to let the UAV recognize dangerous situations and promptly react to them. 

**Challenge**:

- The challenge is however to collect enough data in an unstructured outdoor environment such as a city.
- Clearly, having an expert pilot providing training trajectories is not an option given the large amount of data required and, above all, the risk that it involves for other vehicles or pedestrians moving in the streets. 

**Contributions**:

- Therefore, we propose to train a UAV from data collected by cars and bicycles, which, already integrated into the urban environment, would not endanger other vehicles and pedestrians. 
- Although trained on city streets from the viewpoint of urban vehicles, the navigation policy learned by DroNet is highly generalizable. Indeed, it allows a UAV to successfully fly at relative high altitudes and even in indoor environments, such as parking lots and corridors. 
- To share our findings with the robotics community, we publicly release all our datasets, code, and trained networks.



#### Introduction

**Backgroud:**

- Safe and reliable outdoor navigation of autonomous systems, e.g. unmanned aerial vehicles (UAVs), is a challenging open problem in robotics. Being able to successfully navigate while avoiding obstacles is indeed crucial to unlock many robotics applications, e.g. 
  - surveillance, 
  - construction monitoring, 
  - delivery, 
  - and emergency response [1], [2], [3]. 

- A robotic system facing the above tasks should simultaneously solve many challenges in **perception, control, and localization.** These become particularly difficult when working in urban areas, as the one illustrated in Fig. 1. In those cases, the autonomous agent is not only expected to navigate while avoiding collisions, but also to safely interact with other agents present in the environment, such as pedestrians or cars. 

  <img src=imgs/p3_1.png />

**Traditional methods**:

- **Steps:** The traditional approach to tackle this problem is a **two step interleaved(交错) process** consisting of 
  - (i) automatic localization in a given map (using GPS, visual and/or range sensors), 
  - (ii) computation of control commands to allow the agent to avoid obstacles while achieving its goal [1], [4]. 
- **Disadvantages:**
  - Even though advanced SLAM algorithms enable localization under a wide range of conditions [5], visual aliasing(混叠), dynamic scenes, and strong appearance changes can drive the perception system to unrecoverable errors. 
  - Moreover, keeping the perception and control blocks separated not only hinders(阻碍) any possibility of positive feedback between them, but also introduces the challenging problem of inferring control commands from 3D maps.

**Deep learning Based methods:**

Recently, new approaches based on deep learning have offered a way to **tightly couple perception and control**, achieving impressive results in a large set of tasks [6], [7], [8]. 

- Challenges:
  - Among them, methods based on reinforcement learning (RL) suffer from significantly high sample complexity, hindering their application to UAVs operating in **safety-critical environments**. 
  - In contrast, supervised-learning methods offer a more viable way to learn effective flying policies [6], [9], [10], but they still leave the issue of collecting enough expert trajectories to imitate. 
  - Additionally, as pointed out by [10], collision trajectories avoided by expert human pilots are actually necessary to let the robotic platform learn how to behave in dangerous situations.



**Contributions of this paper:**

Clearly, a UAV successfully navigating through the streets should be able to follow the roadway as well as promptly react to dangerous situations exactly as any other ground vehicle would do. Therefore, we herein propose to use data collected from ground vehicles which are already integrated in environments as aforementioned. Overall, this work makes the following contributions: 

- We propose a residual convolutional architecture which, by predicting the steering angle and the collision probability, can perform a safe flight of a quadrotor in urban environments. To train it, we employ an outdoor dataset recorded from cars and bicycles.
- We collect a custom dataset of outdoor collision sequences to let a UAV predict potentially dangerous situations.
- Trading off performance for processing time, we show that our design represents a good fit for navigation-related tasks. Indeed, it enables real-time processing of the video stream recorded by a UAV’s camera.
- Through an extensive evaluation, we show that our system can be applied to new application spaces without any initial knowledge about them. Indeed, with neither a map of the environment nor retraining or fine-tuning, our method generalizes to scenarios completely unseen at training time including indoor corridors, parking lots, and high altitudes. 

**Remark:**

Even though our system achieves remarkable results, we do not aim to replace traditional “map-localize-plan” approaches for drone navigation, but rather investigate whether a similar task could be done with a single shallow neural network. Indeed, we believe that learning-based and traditional approaches will one day complement each other.



#### Related Work

**Background:**

- A wide variety of techniques for drone navigation and obstacle avoidance can be found in the literature. At high level, *these methods differ depending on the kind of sensory input and processing employed to control the flying platform.* 
- A UAV operating outdoor is usually provided with GPS, range, and visual sensors to estimate the system state, infer the presence of obstacles, and perform path planning [1], [4]. 
- Nevertheless, such works are still prone to fail in urban environments where the presence of high rise buildings, and dynamic obstacles can result in significant undetected errors in the system state estimate. 

**SLAM:**

The prevalent approach in such scenarios is SLAM, where the robot simultaneously builds a map of the environment and self-localizes in it [5]. 

**3D Reconstruction:**

- On the other hand, while an explicit 3D reconstruction of the environment can be good for global localization and navigation, 
- it is not entirely clear how to infer control commands for a safe and reliable flight from it. 



**Learning Based:**

Recently, there has been an increasing research effort in directly learning control policies from raw sensory data using Deep Neural Networks. These methodologies can be divided into two main categories: 

- methods based on reinforcement learning (RL) [7], [11] 
  - While RL-based algorithms have been successful in learning generalizing policies [7], [8], **they usually require a large amount of robot experience which is costly and dangerous to acquire in real safety-critical systems.** 

- methods based on supervised learning [6], [12], [9], [10], [13]. 
  - In contrast, supervised learning offers a more viable(可行) way to train control policies, but clearly depends upon the provided expert signal to imitate. 
  - This supervision may come from 
    - a human expert [6], 
    - hard- coded trajectories [10], 
    - or model predictive control [12]. 

- Disadvantages:
  - However, when working in the streets of a city, it can be both tedious(乏味的) and dangerous to collect a large set of expert trajectories, or **evaluate partially trained policies** [6]. 
  - Additionally, the domain-shift between expert and agent might hinder generalization capabilities of supervised learning methods. Indeed, previous work in [9], [13] trained a UAV from video collected by a mountain hiker but **did not show the learned policy to generalize to scenarios unseen at training time.** 



**Simulation based:**

- Another promising approach has been use simulations to get training data for reinforcement or imitation learning tasks, while testing the learned policy in the real world [14], [15], [11]. 

- Disadvantages:
  - Clearly, this approach suffers from the domain shift between simulation and reality and might require some real- world data to be able to generalize [11]. 
  - To our knowledge, current simulators still fail to model the large amount of variability present in an urban scenario and are therefore not fully acceptable for our task. 
  - Additionally, even though some pioneering work has been done in [14], it is still not entirely clear how to make policies learned in simulation generalize into the real world. 

**Methods of this paper:**

- To overcome the above-mentioned limitations, we propose to train a neural network policy by imitating expert behaviour which is generated from wheeled manned vehicles only. 

- Even though there is a significant body of literature on the task of steering angle prediction for ground vehicles [16], [17], ***our goal is not to propose yet another method for steering angle prediction, but rather to prove that we can deploy this expertise also on flying platforms.*** 
- The result is a single shallow network that processes all visual information concurrently, and directly produces control commands for a flying drone. The coupling between perception and control, learned end-to-end, provides several advantages:
  - such as a simpler and lightweight system 
  - and high generalization abilities. 

- Additionally, our data collection proposal does not require any state estimate or even an expert drone pilot, while it exposes pedestrians, other vehicles, and the drone itself to no danger.



#### Methodology

Our learning approach aims at reactively predicting a steering angle and a probability of collision from the drone on-board forward-looking camera. These are later converted into control flying commands which enable a UAV to safely navigate while avoiding obstacles. 

Since we aim to reduce the bare image processing time, we advocate a single convolutional neural network (CNN) with a relatively small size. The resulting network, which we call DroNet, is shown in Fig. 2 (a). 

***IMPORTANT:***

The architecture is partially shared by the two tasks to reduce the network’s complexity and processing time, but is then separated into two branches at the very end.

- Steering prediction is a regression problem, **while collision prediction is addressed as a binary classification problem.** 
- **Due to their different nature and output range**, we propose to separate the network’s last fully-connected layer. 
- During the training procedure, we use only imagery recorded by manned vehicles. Steering angles are learned from images captured from a car, while probability of collision, from a bicycle.

<img src=imgs/p3_2_1.png />

***A. Learning Approach***

1. The part of the network that is shared by the two tasks consists of a **ResNet-8 architecture** followed by a dropout of 0.5 and a ReLU non-linearity. 
2. After the last ReLU layer, tasks stop sharing parameters, and the architecture splits into two different fully- connected layers. The first one outputs the steering angle, and the second one a collision probability. *Strictly speaking the latter is not a Bayesian probability but an index quantifying the network uncertainty in prediction. Slightly abusing the notation, we still refer to it as “probability”.* 
3. We use mean-squared error (MSE) and binary cross-entropy (BCE) to train the steering and collision predictions, respectively. 

***IMPORTANT:***

Although the network architecture proves to be appropriate to minimize complexity and processing time, a naive joint optimization poses serious convergence problems **due to the very different gradients’ magnitudes that each loss produces.** 

- More specifically, imposing no weighting between the two losses during training results in convergence to a very poor solution. 
- This can be explained by difference of gradients’ magnitudes in the classification and regression task at the initial stages of training, which can be problematic for optimization [19]. 

*二者共用一个前面的参数网络，因为MSE的loss梯度较大，故导致参数更新较大，这反之影响了分类loss的前向，而分类loss较小的梯度反向时几乎无法影响网络参数更新*

到底是两个loss还是一个loss?--应该是一个loss

**Solution:**

- For the aforementioned reasons, imposing no or constant loss weight between the two losses would likely result in sub-optimal performance or require much longer optimization times. This can be seen as a particular form of **curriculum learning** [19]. 

- In detail, the weight coefficient corresponding to BCE is defined in (1), while the one for MSE is always 1. For our experiments, we set $decay = \frac{1}{10}$ , and $epoch_0 = 10$.
  $$
  L_{tot} = L_{MSE} + max(0, 1-e^{decay(epoch-epoch_0)})L_{BCE}
  $$

  - Indeed, the gradients from the regression task are initially much larger, since the MSE gradients’ norms is proportional to the absolute steering error. 
  - Therefore, we give more and more weight to the classification loss in later stages of training. 
  - Once losses’ magnitudes are comparable, the optimizer will try to find a good solution for both at the same time. 

4. The Adam optimizer [20] is used with a starting learning rate of 0.001 and an exponential per-step decay equal to 10−5. We also employ **hard negative mining** for the optimization to focus on those samples which are the most difficult to learn. 

   ？？？？

   In particular, we select the k samples with the highest loss in each epoch, and compute the total loss according to Eq. (1). We define k so that it decreases over time.



**The reason of using Resbolck:**

The residual blocks of the ResNet, proposed by He et al. [18], are shown in Fig. 2 (b). Dotted lines represent skip connections defined as 1×1 convolutional shortcuts to allow the input and output of the residual blocks to be added. Even though an advantage of ResNets is to tackle the vanishing/exploding gradient problems in very deep networks, its success lies in its learning principle. Indeed, the residual scheme has been primarily introduced to address the degradation problem generated by difficulties in networks’ optimization [18]. Therefore, since residual architectures are known to help generalization on both shallow and deep networks [18], we adapted this design choice to increase model performance.

<img src=imgs/p3_2_2.png />

***B. Datasets*** 

To learn steering angles from images, we use one of the publicly available datasets from `Udacity’s project [21]`. 

- This dataset contains over `70,000 images` of car driving distributed over 6 experiments, 5 for training and 1 for testing. 
- Every experiment stores time-stamped images from 3 cameras (left, central, right), IMU, GPS data, gear, brake, throttle, steering angles and speed. 

For our experiment, we only use images from the forward-looking camera (Fig. 3 (a)) and their associated steering angles. 

<img src=imgs/p3_3_1.png />

To our knowledge, there are no public datasets that associate images with collision probability according to the distance to the obstacles. Therefore, we collect our own collision data by mounting a GoPro camera on the handlebars of a bicycle. We drive along different areas of a city, trying to diversify the types of obstacles (vehicles, pedestrians, vegetation, under- construction sites) and the appearance of the environment (Fig. 3 (b)). This way, the drone is able to generalize under different scenarios. 

- We start recording when we are far away from an obstacle and stop when we are very close to it. In total, we collect around 32,000 images distributed over 137 sequences for a diverse set of obstacles. 

- We manually annotate the sequences, so that frames far away from collision are labeled as 0 (no collision), and frames very close to the obstacle are labeled as 1 (collision), as can be seen in Fig. 3(b). 

<img src=imgs/p3_3_2.png />

Collision frames are the types of data that cannot be easily obtained by a drone but are necessary to build a safe and robust system.





C. Drone Control 

The outputs of DroNet are used to command the UAV to move on a plane with forward velocity $v_k$ and steering angle $\theta_k$. More specifically, we use the probability of collision $p_t$ provided by the network to modulate the forward velocity: 

- We use a low-pass filtered version of the modulated forward velocity $v_k$ to provide the controller with smooth, continuous inputs $(0 ≤ \alpha ≤ 1)$:


$$
  v_k = (1−\alpha)v_{k−1} + \alpha (1− p_t)V_{max}
$$






- jfkd



- the vehicle is commanded to go at maximal speed Vmax when the probability of collision is null,
- and to stop whenever it is close to 1.
- 

Similarly, we map the predicted scaled steering $s_k$ into a rotation around the body z-axis (yaw angle θ), corresponding to the axis orthogonal to the propellers’ plane. Concretely, we convert sk from a [−1,1] range into a desired yaw angle θk in the range [−π
2 , π
2 ] and low-pass filter it: θk = (1−β)θk−1 +β
π 2
sk (3)
In all our experiments we set α = 0.7 and β = 0.5, 

while #$Vmax was changed according to the testing environment. The above constants have been selected empirically trading off smoothness for reactiveness of the drone’s flight. As a result, we obtain a reactive navigation policy that can reliably control a drone from a single forward-looking camera. An interesting aspect of our approach is that we can produce a collision probability from a single image without any information about the platform’s speed. Indeed, we conjecture the network to make decision on the base of the distance to observed objects in the field of view. Convolutional networks are in fact well known to be successful on the task of monocular depth estimation [15]. An interesting question that we would like to answer in future work is how this approach compares to an LSTM [22] based solution, making decisions over a temporal horizon.



































# Sonar-Based End to End Learning for a Low-Cost Self-Driving Car

A Low Cost Road Following Task

将图像和超声波数据作为输入（可以当作一个一维图像显示），记录转向，加减速（超声波数据为0加速，有数据减速）和刹车（数据大刹车）

左右超声波数据用来辅助 转向

C

contribution:

Vmax was changed according to the testing environment. women 

## original Dateset

提供有史以来最大的超声波数据集

- esay -baseline
- medium
- hard

### 人类驾驶行为

![](imgs/map.png)

路线：从左下角行驶至右上角

1. 直行阶段：速度打满，角速度基本不调整
2. 进入狭窄区域，减速，主要调整速度，行驶出狭窄区域后左转
3. 稍微加速，然后减速通过狭窄区域
4. 然后加速，沿中线行驶
5. 减速通过狭窄区域
6. 继续直行，然后减速慢慢右转
7. 直行加速



阶段二：

1. 从侧墙边出发，前后无障碍，偏转，然后移动到中线，加速
2. 面对墙，左右无障碍，转入直线，加速
3. 左侧有障碍，前方很远无障碍，大角度转向
4. 从墙边出发，前方有障碍，然后转向



人类驾驶经验：

1. 速度角速度越快，控制难度越大，反之越简单；
2. 转向时应尽量减速，一次转向不可过久，避免回调，回调易引发控制“震荡”
3. 遇到转向过大时应立即减速甚至停车，而不是大角度回调；



初始配置：

1. 速度角速度均为0.3，视线到拐角处开始转向，但不要一次打满，车头即将调正时便要松开转向，不要淞晚了，尽量后期补充也不要回调
2. 失败后停止记录，然后油门为0，开始记录，转向调正后加油门



## test baseline

因为各种可能的组合太多了，首先要通过建立评价指标来一步步选择最优方案

比如是否使用图片比使用原始数据好？

如果使用图片好，图片的尺寸是多大合适？

- 测试集误差能反应实际效果吗？

- 走完全程的时间- 修正一次累积时间

--重要问题：拟合到什么时候是合适的？



基于原始数据分布，概率产生控制量，计算误差



或者先大致训练出一个基本work的模型，看到底哪一个指标对于衡量训练效果是有效的

## image dataset

- 1x16 -原始数据-一维的分类模型
- 16x16 - 仍用一维，确认输入参数增多是否对回归有用
- 16x16 - 图片卷积，确认卷积是否比全连接要好
- 32x16 确定纵轴增加是否有效
  - 如果有效：32x32 确认横轴重复是否有效
    - 如果有效，再次增大 48x48看是否有效
    - 或者每次递增，直到128x128;看评测指标
- 如果无效：16x32

32x32

首先分为

- 训练集
- 验证集：用来模型调参

- 测试集：用来与不同算法比较



### 模型评估

#### 交叉验证

**5次5折交叉验证**

- 将数据集划分为5个大小相似的互斥子集
- 每个子集要保证数据分布的一致性（画出分布图，均值，偏差，方差）--分层采样得到
- 交叉验证
- 随机使用不同的划分重复5次，减小因样本划分不同而引入的差别



## 模型耦合问题

- 两种模型是否分开训练更好？
- Since we aim to reduce the bare image processing time, we advocate a single convolutional neural network (CNN) with a relatively small size.  --如果训练两个模型，时间和参数都会增加
- 如果分开训练，就能较好的何时结束训练的问题；
- 关键是转向动作与速度控制是否是强耦合的，二者能否分开来看
  - 应该不能，同样的转弯动作，我角速度大了，速度就要大一些，角速度小，速度也要小一些，甚至停止等待转向完成；而不是仅仅转弯时减速就可以解决
- 或者只要数据集够大就可以？



或者采用角速度与一个刹车系数，在基础速度上加速减速，这样大部分是1也可以保证车辆是运行的

还有损失函数的定义应该也很重要

现在不好训练我觉得还是因为数据集数据分布的问题



while collision prediction is addressed as a binary classification problem.

- 分开训练，采用不同的全连接层；多个全连接层；
- 如果没效果，也改成二分类问题；加速，减速，
- 观察训练过程，自己定义loss function -- 要仔细分析损失函数是如何变化的；以及参数更新过程
- 还是要自己定义loss, 因为你不知道你传入的标签列表是如何定义损失的



## 新模型

1. 直接输入16个深度，采用多层感知机
2. 16个深度，采用LSTM
3. 图片，卷积



首要的还是要好好处理数据集；如何解决同时训练速度角速度的问题；主要还是由于的速度是否有一定的规则，如果没有，将很容易干扰角速度的训练；

- 采集时固定速度，只拟合角速度
- 尝试固定速度，速度是一个分类，只有加速，减速两个选项；速度是一个分类问题；



1x1卷积升维 densenet

全部1x1卷积然后全连接

1x1 与3x3



其实也不用纠结，直接用最好的模型，然后在此模型上改进



## 比较

用原始模型和我的模型比较还是用类似模型和我的模型比较？

- 图片输入不匹配，是重复扩大还是reseize还是

还是要根据前面的实验结果来确定





年少不知曲中意， 再听已是曲中人。
曲终人散梦已醒， 梦中之人何处寻。
回首已是曲中人， 何须在意曲中意。
莫非在等梦初醒， 再去寻那梦中人。
不愿再做曲中人， 奈何越听越沉沦。
既然已成曲中人， 何必在听曲中曲。
曲中人唱曲中曲 ，曲中人非曲中人。