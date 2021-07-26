# Neural_Network_Charity_Analysis

## Overview of the Analysis
Using Machine Learning and Neural Networks for this project, I used the features in the dataset to create a binary classifier that will help to predict if the applicants that will be funded by a Charitable organization called Alphabet Soup will be successful. For this analysis we had a dataset containing various measures on 34,000 organizations that have been funded by Alphabet Soup. This project compromised of the following 3 steps:

•	Preprocessing the data for the neural network

•	Compile, Train and Evaluate the Model

•	Optimizing the model

## Results

### Data Preprocessing

•	The columns EIN and NAME are identification information and have been removed from the input data.

•	The column IS_SUCCESSFUL contains binary data refering to weither or not the charity donation was used effectively. This variable is then considered as the target for our deep learning neural network.

•	The following columns APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features for our model.

•	Encoding of the categorical variables, spliting into training and testing datasets and standardization have been applied to the features.

### Compiling, Training, and Evaluating the Model

•	This deep-learning neural network model is made of two hidden layers with 80 and 30 neurons respectively.

•	The output layer is made of a unique neuron as it is a binary classification.

•	I used the activation function ReLU for the hidden layers and the output is a binary classification, Sigmoid is used on the output layer.

•	 ![image](https://user-images.githubusercontent.com/78935982/126924063-05ac11e0-6e87-445e-a6eb-52fded60802a.png)


•	The optimizer is Adam and the loss function is binary_crossentropy.

•	 ![image](https://user-images.githubusercontent.com/78935982/126924084-177d427f-cbad-4f38-ba92-1b329207e969.png)


•	The model accuracy is under 75%. 

•	I increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.

•	 ![image](https://user-images.githubusercontent.com/78935982/126924094-b31c5189-3c51-4f11-8e19-f08eb6da2f6d.png)


•	Then I tried a different activation function (tanh) but none of these steps helped improve the model's performance.

•	 ![image](https://user-images.githubusercontent.com/78935982/126924107-3cabf942-2a6f-454d-845b-bd8aebad336d.png)


## Summary
The deep learning neural network model did not reach the target of 75% accuracy. Considering that this target level is pretty average we could say that the model is not outperforming. Since we are in a binary classification situation, we could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against our deep learning model.
