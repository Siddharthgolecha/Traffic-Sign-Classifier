## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
One of the things, Autonomous Vehicles should be good at is recognising the Traffic Signs across the street. The Traffic Signs give instructions or provide information to drivers and road users. They represent rules that are in place to keep one safe, and help to communicate messages to drivers and pedestrians that can maintain order and reduce accidents.

So the recognition and the classification of the Traffic Signs become so important for the Autonomous Vehicles. In this project, I used what I've learned about deep neural networks and convolutional neural networks to classify traffic signs using one of the simplest CNN architecture model and the one of the first used for this specific purpose too, i.e Lenet-5. I trained my model over the German Traffic Sign Data set and achieved Training accuracy of 99.18% , Validation accuracy of 99.93% and Test accuracy of 94.4%. After the model is trained, I then tried out my model on 5 random images of German traffic signs that I found on the web to check its accuracy and it correctly classified them all.  

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

Download the data set. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set. You can also download from from here and split it accordingly. [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

