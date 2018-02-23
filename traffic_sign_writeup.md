
# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

Code for this in Cell 3. It shows frequency for input Label for corresponding dataset (Training, Testing, Valid).
![training.png](attachment:training.png)
![testing.png](attachment:testing.png)
![valid.png](attachment:valid.png)

Code for this in Cell 3. It shows random 45 image from training dataset.
![input_image.png](attachment:input_image.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The code for this is in cell 4.
First I have changed the image in YIQ model as it gets you the color information (in the IQ or CbCr components) in a smaller amount of data than is the case in RGB. Then changed channel to 1 and at last scaled the pixel from [-1,1] as it will improve the performance while training the data. Image will look loke below after changing it to YIQ           ![preprocessing_sample.png](attachment:preprocessing_sample.png)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for this in cell 7.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image      							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU           		|           									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  			    	|
| Fully Connected		|												|
| RELU					|												|
| Dropout				| Keep Prob : 0.8								|
| Fully Connected		|												|
| RELU					|												|
| Fully Connected		|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
Code is in cell 11.
To train the model, I used an LeNet for the most of the part that was given in course but I have used dropout in it with keep_prob=0.8. I have used AdamOptimizer, lerning rate=0.002, Batch_size=128, Epochs=80. Apart from this I have taken out 20% from training dataset and used it to calculate accuracy at each epoch to get currect accuracy at the end. Then finally I used the given validation set for calculating final validation.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Training and validation accuracy was calculated in cell 11 and test accuracy was calculated in cell 12.

My final model results were:

training set accuracy of 99.9%

validation set accuracy of 96.8%

test set accuracy of 94.5%

I had first trained the model in Lenet architecture without changing anything but it didn't met my expectation then I normalised it and added a dropout in it. After tuning the keep_prop and learning rate many times, it met the expectation.

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? 

I used the architecture which was explained in video.

* What were some problems with the initial architecture? 

Lack of knowledge was the first issue. After fixing it LeNet architecture worked well with defaults.

* How was the architecture adjusted and why was it adjusted? 

Adjusted the learning rate and adding dropout helped a lot.

* Which parameters were tuned? How were they adjusted and why? 

Epoch, learning rate, and drop out probability were all parameters tuned. Firsst i had trained the model for less epoch and less dropout. Then slowly I increased the epoch and simultaneouly drop out to reach to my expextation. As I reached the expectation I changed the learning rate so that I can reach my expectation in lesser time. 

* What are some of the important design choices and why were they chosen? 

I think I can take this project ahead and learn a lot in this field. The important design choice was having a uniform dataset with enough convolutions to capture features with good accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used semi-easy images to classify and and modified them somewhat. I made them uniform in size and some image partly chopped of that could be difficult to classify. Fourth image was misclassified as it is somewhat tilted and lack in the quality.![Image1.jpg](attachment:Image1.jpg)![Image2.jpg](attachment:Image2.jpg)![Image3.jpg](attachment:Image3.jpg)![Image4.jpg](attachment:Image4.jpg)![Image5.jpg](attachment:Image5.jpg)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for this is in 15 cell.
The model was able to guess 4 out of 5 traffic signs correctly, which gives 80% accuracy.This compares favorably to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Turn right ahead  	| Turn right ahead								|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Speed limit (70km/h)  | Speed limit (20km/h)			 				|
| Stop       			| Stop               							|

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for this is in 15 cell.
For the first image, the model is relatively sure that this is a Road work (probability of 1.0), and the image does contain a Road work. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Road work   									| 
| 2.96407001e-37     	| Dangerous curve to the right 					|
| 0.00000000e+00		| Speed limit (20km/h)						    |
| 0.00000000e+00	    | Speed limit (30km/h)					 		|
| 0.00000000e+00	    | Speed limit (50km/h)      					|

For the second image, the model is relatively sure that this is a Turn right ahead (probability of 1.0), and the image does contain a Turn right ahead. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Turn right ahead   							| 
| 6.26136764e-33     	| No passing for vehicles over 3.5 metric tons	|
| 1.56481418e-34		| Ahead only         						    |
| 0.00000000e+00	    | Speed limit (20km/h)					 		|
| 0.00000000e+00	    | Speed limit (30km/h)      					|

For the third image, the model is relatively sure that this is a Speed limit (30km/h) (probability of 1.0), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Speed limit (30km/h)   						| 
| 8.27261521e-20     	| Speed limit (20km/h)	                        |
| 1.81916373e-23		| Speed limit (80km/h)         				    |
| 5.20136276e-25	    | Speed limit (50km/h)					 		|
| 9.34450039e-28	    | Stop      				                 	| 

For the fourth image, the model is relatively sure that this is a Speed limit (20km/h) (probability of 0.9), and the image does contain a Speed limit (70km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.98623252e-01        | Speed limit (20km/h)   						| 
| 1.37673574e-03     	| Speed limit (70km/h)	                        |
| 1.55087705e-20		| General caution         			     	    |
| 2.16491377e-22	    | Roundabout mandatory			 		 		|
| 2.16048023e-25	    | Speed limit (30km/h)      			       	| 

For the fifth image, the model is relatively sure that this is a Stop (probability of 0.9), and the image does contain a Stop. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.95050967e-01        | Stop   						                | 
| 3.00390669e-03     	| Traffic signals	                            |
| 1.82864070e-03		| General caution         			     	    |
| 4.38175121e-05	    | Turn right ahead			 	     	 		|
| 3.92783659e-05	    | Bumpy road                			       	| 


