# Predicting Star Rating of Reviews on Yelp based on Sentiment Analysis of Annotated Reviews Data Set

## Introduction
Yelp reviews consist of a star rating (lowest number indicates poor and highest number indicates excellent) as well as text reviews. In this project, we will be predicting the star rating corresponding to a text review. 

## Preprocessing Data
The data has to be cleaned and preprocessed first before feeding to our training model. NLTK's stopwords are used to eliminate the most commonly used words. Only reviews in English are considered. The selected reviews are converted to lowercase and special characters are removed.

Command to run preProcess file. It takes two arguments as input i.e. business.json and review.json:

python3 preProcess.py business.json review.json 

Expected Output will be a processed csv file with name "processedFile.csv"

processedFile.csv


## Model 

We have comon file to either train mode or test model or both. It takes our processedFile.csv and the type what you want to do with file.

### Training Model
GRU is used for training the model with AdamOptimizer. Word embeddings for the model are taken from the open dataset Numberbatch available from ConceptNet- https://github.com/commonsense/conceptnet-numberbatch. 

Command to run model file for training. Type is "train"

python3 model.py train

Expected Output will be a folders inside which model will be saved.

### Testing Model

Command to run model file for testing. Type is "test"

python3 model.py test

Expected Output will be a Accuracy printed and you should inside which model will be saved. 


## Output

Test accuracy is: 0.5645978009259259
