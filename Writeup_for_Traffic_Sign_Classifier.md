# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Training_Sample_Histogram.PNG "Visualization Of Training Samples"
[image2]: ./examples/Validation_Sample_Histogram.PNG "Visualization Of Validation Samples"
[image3]: ./examples/Test_Samples_Histogram.PNG "Visualization Of Test Samples"
[image4]: ./examples/Sample_Training_Images.PNG "Sample Training Images"
[image5]: .examples/germanwebimg_1.jpg "Traffic Sign - Roadwork"
[image6]: ./examples/germanwebimg_2.jpg "Traffic Sign - Slippery Road"
[image7]: ./examples/germanwebimg_3.jpg "Traffic Sign - Right-of-way at the next intersection"
[image8]: ./examples/germanwebimg_4.jpg "Traffic Sign 4 - Speed limit (60km/h)"
[image9]: ./examples/germanwebimg_5.jpg "Traffic Sign 5 - Children crossing "

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/LakshmiPrasannan/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in Test, Validation and Test Image samples.

![alt text][image1]

It is observed in the training sample that highesht number of samples are constituted by speed and direction signals with around samples count ranging between 1750 and 2000, whereas general caution samples (code-18) are moderate.

![alt text][image2]
It is observed in the validation sample that highest the same trend is followed proportionally as that of training sample.


![alt text][image3]
In the test sample though speed and direction signals still contribute the majority, unlike the training and validation signal we can see a higher proportion of General Caution signals compared to train and validation samples.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For the pre-processing of data I had adopted 3 strategies.
1. Shuffle the training data - This helped in the training model being unbiased upon the order of the images. 
2. Normalizing the X_data - 
When the images in the training datasets were displayed, some of the images were found to have different intensities of contrast and brightness as in the images with code 31 and 3 in the following images.

![alt text][image4]
So hence I realized it would be ideal to bring all the images to uniform pixel intensities using normalization, so that the pixel intensities would range in between 0 and 1. The normalization was applied to all training, validation and test data.









#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer    5 Layes 		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| Activation 			| ReLU      									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 5 Layes , output = 43.   					    |
| Fully connected		| Yes        									|
| DropOut				| Yes        									|
|						|												|
|						|												|
 
 The LeNet architecture was defined using Convolutional Neural Networks, that had 5 layers. 
 Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
 Activation: ReLu
 Pooling. Input = 28x28x6. Output = 14x14x6.
 Stride: 1x1
 
 Layer 2: Convolutional. Output = 10x10x16.
 Activation: ReLu
 Pooling. Input = 10x10x16. Output = 5x5x16.
 Stride: 2x2
 
 Layer 3: Fully Connected. Input = 400. Output = 120.
 Activation: ReLu
 Dropout implemented 
 
 Layer 4: Fully Connected. Input = 120. Output = 84.
 Activation: ReLu
 Dropout implemented 
 
 Layer 5: Fully Connected. Input = 84. Output = 43.
 Activation: ReLu
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
The training pipelines are avaliable in the 11, 12 and 13 cells
Before training the model the following hyperparameters are defined in cell 7 and 11
EPOCHS = 40
BATCH_SIZE = 128
and 
rate = 0.001 , which is the learning rate
beta = 0.071 , which is the multiplying factor for L2 Regularization.

Some variable placeholders are defined in cell 10
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

Also the classes are modified to one_hot format in cell 10
one_hot_y = tf.one_hot(y, 43)
And in the same cell the keep_prob for Drop out is also defined
keep_prob = tf.placeholder(tf.float32)
To train the model, I used a convolutional neural network that was trained for the specific epochs with the mentioned batch size of data. Once the network was defined at each step the following were found
1. Cross Entropy using softmax_cross_entropy_with_logits of the tensor flow
2. Using the cross entropy reduced_mean function of tensor flow was used to extract the loss operation.
3. To prevent overfitting of data, onto the existing loss_operation as regularizer value was added. The regularizer value was a scalar multiplied to the previous weight of the network layer.
4. In order to optimize the model AdamOptimizer was used so that the model was trained in considerable time period with further accuracy.
5. Once the training model was ran, optimizer minimize operation was executed over the loss function to reduce the loss in each iteration.

6. The training function was called setting proper offset values in batches for specific number of epochs.
7. The trained results were further made to run through an evaluate model to find the validation accuracy.





#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95.7%
* validation set accuracy of 95.6%
* test set accuracy of 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The architecture chose was implementing LeNet() using Convolutional Neural Network.
As we were training images, I realized the best iterative method to classify images are using layered network, that scans the image using multiple strides in each layer. This will ensure the whole image being learnt over multiple learnings in one iteration itself and the chances of wrong prediction is reduced considerably.

* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? 
Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The following adjustments were implemented in the architecture to avoid overfitting

###### Implementing L2 Regularization:
 
Understanding the considerably high variance in the count of sample images among many of the sign images, I realized an L2 Regularization would prevent an overfit of the training model on the test. So an L2 regularization was introduced where in a regularization factor with beta multiplied by weights were added to the loss operation.

###### Implementing Droput:
However with only L2 regularization, the training model was achieving only a test accuracy of less than 91% with an Epoch size of 40. As increasing epochs wasn't found to be a great idea considering the increase time required for computing, I decided to introduce some dropouts where in the keep_probability was set to be 0.6 for training data and 0.1 for validation model.
 

* Which parameters were tuned? How were they adjusted and why?

The Epochs and beta ( regularization parameter) were the parameters that were tuned.

With and Epoch of 10, it was found the the model lacked accuracy and more iterative learning was recommended for higher accuracy.

Learning rate and Batch_Size were maintained at 0.01 and 128 respectively as increasing these parameters ended up in overfit of the the data and decreasing resulted in underfit.

The regularization parameters was tuned to 0.071 to give a considerable proportion to the previous weights so that it prevented overfit of the training sample.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolutional neural networks have the property of using iterative learning methods using strides. This will help the learning to be iterative giving ample weightage to the previously learnt layer as the network appends previously learnt information from 1st layer to the next. The feature recognition is never attained in a one go here. Eg. the image is divided into many parts using the strides and the combined data can only give the whole picture and thereby full information of the picture. This learning that happens layer by layer is more effective in image recognition.


* Why did you believe it would be relevant to the traffic sign application?
As traffic signs were images provided on various shapes of sign boards a general image classification will not help in identifying the sign. Since most of the Sign board's useful information lies towards the centre of every image, I believe Convolutional Neural Networks are best in extracting the central information of the image and identifying the information passed in the sign board.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9]


These images are resized to match their size of 32x32x3.
The second image might be difficult to classify because the count of samples of Slippery Roads are very less in the testing samples, hence they are often detected as General Caution or Speed sign's images whose count is considerably higher than slipper roads in the testing sample.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                       |     Prediction	        					| 
|:------------------------------------:|:------------------------------------------:| 
| Children crossing              	   | Children crossing   						| 
| Speed limit (60km/h)          	   | Speed limit (60km/h) 						|
| Slippery road          			   | Speed limit (120km/h) 						|
| Road work                 		   | Road work         			 				|
| Right-of-way at the next intersection| Right-of-way at the next intersection 		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.6% accuracy

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Road work sign (probability of 0.99987), but the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	                                					| 
|:---------------------:|:---------------------------------------------------------------------:| 
| 0.99987      			| Road work                         									| 
| 0.0001  				| Double curve                   										|
| 0.00002  				| Traffic signals                   									|
| 0.        			| Beware of ice/snow                             		 				|
|0.     				| Right-of-way at the next intersection        							|


For the second image, the model is relatively sure that this is a Speed limit (120km/h) sign (probability of 0.96098), but the image does contain a Speed limit (120km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	                                					| 
|:---------------------:|:---------------------------------------------------------------------:| 
| 0.96098      			| Speed limit (120km/h)             									| 
| 0.02994  				| Bicycles crossing              										|
| 0.00718  				|Right-of-way at the next intersection									|
| 0.0005       			|Slippery road                                   		 				|
| 0.00045  				|Beware of ice/snow                            							|

For the third image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 0.1), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	                                					| 
|:---------------------:|:---------------------------------------------------------------------:| 
| 1.        			| Right-of-way at the next intersection									| 
| 0.    				| Beware of ice/snow            										|
| 0.      				|Double curve               											|
| 0.         			|End of no passing by vehicles over 3.5 metric tons		 				|
| 0.     				|Pedestrians                                  							|


For the fourth image, the model is relatively sure that this is a Speed limit (60km/h) sign (probability of 0.92225), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	                                					| 
|:---------------------:|:---------------------------------------------------------------------:| 
| 0.92225      			| Speed limit (60km/h)               									| 
|0.07664 				| Speed limit (50km/h)          										|
|0.00111				| Speed limit (80km/h)         											|
|0.         			| Dangerous curve to the right             				 				|
| 0.     				| Stop                                       							|


For the fifth image, the model is relatively sure that this is a Children crossing sign (probability of 0.7023), and the image does contain a Children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	                                					| 
|:---------------------:|:---------------------------------------------------------------------:| 
| 0.7023       			| Children crossing                 									| 
| 0.07965  				| Dangerous curve to the right     										|
|0.0604					| Bicycles crossing            											|
|0.05671 	   			| Ahead only                         					 				|
|0.02461				| Beware of ice/snow                         							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

#### References Used :
For implementing L2 Regularization using TensorFlow
https://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/



