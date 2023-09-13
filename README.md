# Movie Genre Prediction
## Purpose
The purpose of this project is to create a neural network model that can predict a movie genre based on the description. I completed this project to explore Natural Language Processing (NLP), which is a machine learning technology that gives computers the ability to interpret, analyze, manipulate and comprehend human language. The process for completing this project includes:

(1) Data Cleaning/Preparation

(2) Data Splitting

(3) Model Selection/Architecture

(4) Creating Predictions and Model Evaluation


## Overview:
Environment: Jupyter Notebook

Language(s): Python

Dependencies: pandas, numpy, matplotlib, nltk, tensorflow

## Project Steps
### Data Cleaning/Preparation:
In preparation for testing and training a neural network model, I balanced the data. To balance the data I analyzed the distribution of data based on Genre category. I found that a majority of the data fell within three genres (Drama, Documentary, and Comedy). I then filtered the dataset by these categories, ensuring there was a similar number of examples for each genre. 

The next step in this project was to clean and preprocess the data. I removed stopwords, special characters and converted descriptions to all lowercase to reduce noise and irrelevant information. I used the pre-processing technique of lemmatization to break words down to their root meaning. I also used the technique of tokenization to split paragraphs and sentences into smaller units.


### Data Splitting
Once the data was cleaned and processed, I split the data into training and testing sets, 80-20. The training set is used to train the model and the testing set is used to evaluate the performance of the model.


### Model Architecture
I tested a variety of model architectures using tensorflow to try to optimize the models performance.

The first layer is an embedding layer to turn text into numerical values, allowing the model to work with text data and teach it to understand the meanings.

The second layer is Global Average Pooling 1D. This layer is commonly used in sequence-based tasks, such as text classification and sentiment analysis. GAP-1D helps the model focus on the critical information in the text and classify it.

The third layer is a Dense layer using the ReLU activation function. The Rectified Linear Unit Activation is a hidden layer in deep neural networks commonly used for natural language processing.

The last layer is a Dense layer using the Softmax activation function. This activation is used in multi-class classification problems by converting raw scores (logits) into class probabilities. 


### Creating Predictions and Model Evaluation
I used the following metrics to analyze how well my model was at predicting movie genre based on the description.
* Loss: the error between predicted and actual values, lower loss values indicate better model performance. 
* Accuracy: the proportion of correctly classified values out of the total number in the dataset. Higher accuracy is better.
* Confusion matrix: breaks down the models predictions and how they compare to the actual values.
* Precision: measures the proportion of correctly predicted positives out of all positive predictions made by the model. High precision means that when the model predicts a positive class, it's likely to be correct.
* Recall: measures the proportion of true positive predictions out of all actual positive samples in the dataset. High recall means that the model correctly identifies a large portion of the actual positive samples.
* F1-score: balanced metric that combines both precision and recall into a single value. Ranges from 0 to 1, higher values indicate better model performance in terms of both precision and recall. 


#### Loss
As the model trains over the number of epochs specified, the loss value for both the training and testing sets decreased. This is a positive sign that the loss value decreased, however it is still high with a value of .71 for the training set and .69 for the testing set. Meaning the models predictions are not very close to the true values. 

#### Accuracy
The accuracy of both the training and testing sets increased after each epoch, the last epoch had an accuracy of 68% for the training set and 69% for the testing set. This indicates that the model was able to learn after each epoch and improve performance. 

#### Confusion Matrix
Looking at the confusion matrix, the model did not predict any of the descriptions as a Comedy. This leads me to believe that the data is not evenly distributed and there are too few Comedy values in the dataset for the model to learn from. Most comedies were incorrectly predicted as Drama's.

Looking at the actual and predicted labels for Dcoumentary and Drama, the model has a high number of true positives for each.

#### Precision
The precision for Documentary is highest .82, meaning the model is successful at avoiding false positives. The precision for Drama is .60, this is confirmed by the Confusion Matrix as the model predicted a high number of Comedies as Dramas.

#### Recall
The recall score for Documentary is .90  and Drama is .86, this indicates the model's ability to identify true positives is high.

#### F1-score
The F1-score for Documentary is .86 and Drama is .71.

## About the Dataset
Data: Genre Classification Dataset IMDb

Source: Kaggle

Files: description.txt, test_data_solution.txt, test_data.txt, train_data.txt


