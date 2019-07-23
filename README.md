# Prediction-of-Hospital-Readmission
The aim of this project is able to identify risk of unplanned hospital readmission of a patient in the next 30 days after his discharge from the hospital 

# Healthcare Analytics Problem - 
Task(T): Classify whether a patient get readmission?
Experience: Available data of past behaviour of patient related to readmission.
Performance(p): F1_score 

# Dataset Used -
UCI is publicaly available dataset
It contains data about de-identified diabetes patient encounters for 130 US hospitals (1999–2008) 
It contains 101,766 observations over a period of 10 years. 

# Data Pre-processing: 
# Dealing With Missing Values: 
Weight, Payer_code and medical_speciality was discarded because number of missing values was so high. In Race,  missing values replaced by mean.and in Diagnosis 1, Diagnosis 2 and Diagnosis 3,  we replace those missing values which is common in these three.
# Categorization of Diagnosis:
The dataset contained upto three diagnoses for a given patient (primary, secondary and additional). However, each of these had 700–900 unique ICD codes and it is extremely difficult to include them in the model and interpret meaningfully.

# Collapsing Some Other Variables :
Emergency
Urgent
Elective
Newborn                               collapse into 1 label
Not available  
Null
Trauma center
Not mapped

# Transformation :
Upwardly skew data is very general in health-care analytic. We know upwardly skew data is log normally distributed. In machine learning problems we believe on normalized data .we can normalize upwardly skew data by taking logs of data.If data is downwardly skew than logging will not work and not give good results. downwardly skewed data can be corrected by taking the square , cube and anti-log of data.All of these transformation change the magnitude of the distribution . Transformation never change the relative change of the order of value.


# Correlation :
We see the effect of each column by calculate the correlation matrix. Apply multiplication operation between highly correlated feature that gives some sense.



# Dealing with un-balanced Data:
3 algorithms, Adasyn, Smote-ENN and Smote were tried for data balancing.
combine over- and under-sampling using SMOTE and edited nearest neighbours.

# MODELING 

Hyperparameter Tuning: 

A machine learning model is just a formula with a number of parameters that need to be learned from data. But there is parameter that can’t be learned from the regular training process is called hyperparameter.
Two techniques are used for parameter tuning  one is manual and second is by gridsearchcv. Cross Validation used by me.


Cause of the Poor Performance:

It maybe because of overfitting and underfitting. 

In my project, the model shows overfitting.
Overfitting: A hypothesis h is said to overfit the training data if there is another hypothesis h’ s.t. h’  has more error than h on training data but h’ has less error than h on test data.
Overfitting occurs when model is too complex.


Limit overfitting:

There are two techniques to limit overfitting.
Use resampling technique to estimate model accuracy.
Hold back to a validation dataset.
Regularization

Cross Validation:
The general procedure is as follows:
Shuffle the dataset randomly.
Split the dataset into k groups
For each unique group:
Take the group as a hold out or test data set
Take the remaining groups as a training data set
Fit a model on the training set and evaluate it on the test set
Retain the evaluation score and discard the model
Summarize the skill of the model using the sample of model evaluation scores

Reducing Overfitting in RF-model: 
N_estimators - The lower the number , the closer the model is to a decision tree with a restricted feature set.
Max_feature - This determines how many feature each tree is randomly assigned, The smaller the less likely to overfit, but too small will start to introduce 			under-fitting.
Max_depth - Experiment with this, This will reduce the complexity of the learned models learning overfitting risk.
Min_sample leaf - Try setting this to value greater than one it means the branch will stop splitting once the leaves have that number of samples each.

Reducing Overfitting in the Neural Network:
Dropout - Dropout deletes random sample of the activation in the training. A good practice is to start with a low dropout in the first layer and then gradually increase it.
Early Stopping technique - We stop when the performance has stopped improving on hold-out validation set.

Convolutional Neural Network (CNN):
Relu activation function is used for hidden layer and softmax for output layer. Optimizer was chosen Adam. But the result obtained does not satisfies the model.

# Deployment on Flask framework:

Python pickle module is used for serializing and de-serializing a Python object structure. 
Any object in Python can be pickled so that it can be saved on disk. 
Pickle “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. 
The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

This function wraps dumps() to add a few enhancements that make life easier. It turns the JSON output into a Response object with the application/json mimetype.

Flask.json.dumps - serialize object into a json formatted
Flask.json.load - unserialize a json obj from a string 

To access the incoming data in Flask, you have to use the request object. The request object holds all incoming data from the request, which includes the mimetype, referrer, IP address, raw data, HTTP method, and headers, among other things.








