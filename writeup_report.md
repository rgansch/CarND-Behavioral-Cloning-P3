#**Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py encapsulated class to create and train the model
* nn\_arch.py containing the neural network architecture (used by model.py, idea was u easily switch architectures)
* image\_gen.py generator class that supplies images from several recording sets for training
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py and nn\_arch.py files contain the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a network I found on the internet: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ which has been developed by nvidia for automated driving. It was modified with dropout layers to prevent overfitting.
- Cropping2D to remove upper and lower regions of images
- Lambda function with normalization
- Conv2D 24x5x5 with LeakyReLu activation
- Conv2D 36x5x5 with LeakyReLu activation
- Conv2D 48x5x5 with LeakyReLu activation
- Conv2D 64x5x5 with LeakyReLu activation and dropout
- Conv2D 64x5x5 with LeakyReLu activation and dropout
- Dense 1164 with LeakyReLu activation
- Dense 100 with LeakyReLu activation
- Dense 50 with LeakyReLu activation
- Dense 10 with LeakyReLu activation
- Dense 3

The final 3 output parameters correspond to angle, throttle and brake actuation.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers and shuffling of training data in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road:
The data sets were recorded in several runs on track 1:
- set1: 1 lap clockwise
- set2: 1 lap clockwise
- set3: 1 lap clockwise
- set4: 1 lap counter-clockwise
- set5: 1 lap counter-clockwise
- set6: 1 lap counter-clockwise
- set7: recovery maneuvers clockwise (e.g. driving off-center to center)
- set8: recovery maneuvers counter-clockwise
- set9: problematic maneuvers identified after several validation runs (curves with small radius mostly)

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use an existing model known to perform well and adapt it to the situation.

In order to gauge how well the model was working, I split my image and maneuvering data into a training and validation set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (curves with small radius) to improve the driving behavior in these cases, I added the set9 and retrained the model with this set till the result was satisfactory.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Since my notebook running the simulator runs into performance problems the car sways from left to right quite a bit. This was especially evident when recording the images for the video. Without recording the car ran on a more stable course.
Since the second track seemed much more taxing computation wise I didn't try training model there. My notebook would most likely not be able to provide the inputs sufficiently fast.

####2. Final Model Architecture

The final model architecture didn't not change to the initial one, since it performed very well from the start.

####3. Creation of the Training Set & Training Process

To obtain a larger training set the images from the center camera were added as mirrored images with negative steering angle. The images from the left and right camera were added as well with a steering offset of 0.09.

###Test in autonomous mode
The result of the final test in autonomous mode can be watched in track1.mp4
