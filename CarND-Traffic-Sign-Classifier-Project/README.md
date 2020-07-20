# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/original_images.png "Original images"
[image2]:./images/visualization.png "visualization"
[image3]: ./images/comparision.png "comparision"
[image4]: ./images/preprocessing.png "preprocessing"
[image5]: ./images/5x.png "Traffic Sign 1"
[image6]: ./images/3x.png "Traffic Sign 2"
[image7]: ./images/4x.png "Traffic Sign 3"
[image8]: ./images/2x.png "Traffic Sign 4"
[image9]: ./images/1x.png "Traffic Sign 5"
[image13]: ./images/top_5.png "Top 5 predictions"
[image14]: ./images/softmax.png "Top 5 softmax probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Annd I would like to thank StackOverflow for providing me solutions to the problems I was facing and the related code snippets too sometimes.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

```
import numpy as np

# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_valid)

# Number of testing examples
n_test = len(X_test)

# Shape of an traffic sign image
image_shape = X_train[0].shape

# Unique classes/labels in the dataset
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![8 random images from training data set][image1]

It is a bar chart showing how the data ...

![Bar chart][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I experimented with different types of color schemes like grayscale, HSV and HLS. However, after experimenting with different color schemes and even merging them, I decided to stick with grayscale only.

Here is an comparasion with the grayscale, Lightness of HLS and Value of HSV as well

![Grayscale and HSV comparision][image3]

Here is an example of a traffic sign image before and after grayscaling.

![After Grayscale][image4]

As a last step, I normalized the image data because the mean was around 80 and to bring the mean around to improve the accuracy, normalization was neccessary which resulted in mean to reduce to around -0.17.   

I augmented the data with the help of ImageDataGenerator of Keras.
```
datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1)
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		| Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
|  Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32				|
|  Convolution 5x5     	| 1x1 stride, valid padding, outputs 12x12x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x48 				|
|  Flatten     	| Inputs 6x6x48 , outputs 1728 	|
|  Fully connected		| 			Inputs 1728 , outputs 256				|
| RELU					|
|  Fully connected		| 			Inputs 256 , outputs 84				|
| RELU					|												|
|  Fully connected		| 			Inputs 84 , outputs 43				|
| Softmax				| outputs 43       									|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used Adam optimizer with the learning rate = 0.0005 and the loss function as Categorical Cross Entropy. I used the ReduceLROnPlateau function by Keras to reduce the learning rate whenever it encounters, a plateau for validation accuracy. I then used the ImageDataGenerator to augment images to provide different test cases. I set the batch size of 100 and number of epoches to 40.

```
model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.2, min_lr=1e-6)
datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1)
datagen.fit(X_train_normalized)
history = model.fit_generator(datagen.flow(X_train_normalized, y_one_hot, batch_size=100),\
                    steps_per_epoch=len(X_train_normalized)/100,
                    epochs=40, validation_data = (X_valid_normalized, y_one_hot_valid), callbacks=[reduce_lr])
model.save("lenet5")


```
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 99.18%
* Validation set accuracy of 99.93%
* Test set accuracy of 94.40%

If a well known architecture was chosen:
* What architecture was chosen?
 * I chose the Lenet5 Architecture which is a pretty basic classifier and one of the first classifiers used for traffic sign Recognition.
* Why did you believe it would be relevant to the traffic sign application?
 * It was already used and one of the classifier for traffic sign Recognition.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 * The loss funnction for the validation data accuracy and training data accuracy were about tho converge which means the model was just right. It was neither underfitting nor overfitting. And the test result proved that by giving an accuracy of 94.4%.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 5 German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]

I actually chose all the images randomly, which I thought will be difficult.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Keep Right      		| Keep Right  									|
| Priority Road     			| Priority Road										|
| Speed Limit(60 km/h)	      		| Speed Limit(60 km/h)	 				|
| Speed Limit(30 km/h)			| Speed Limit(30 km/h)					|
| Right-of-way at the next intersection  | Right-of-way at the next intersection |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 12630.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model was absolutely certain with the predictions and here is the image showing the probabilities of all the test images.

![Top 5 Predictions][image13]

![Top 5 Probabilities][image14]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
