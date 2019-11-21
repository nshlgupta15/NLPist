# Predicting Star Rating of Reviews on Yelp based on Sentiment Analysis of Annotated Reviews Data Set

## Introduction
Yelp reviews consist of a star rating (lowest number indicates poor and highest number indicates excellent) as well as text reviews. In this project, we will be predicting the star rating corresponding to a text review. 

## Preprocessing Data
The data has to be cleaned and preprocessed first before feeding to our training model. NLTK's stopwords are used to eliminate the most commonly used words. Only reviews in English are considered. The selected reviews are converted to lowercase and special characters are removed.

## Training Model
GRU is used for training the model with AdamOptimizer. Word embeddings for the model are taken from the open dataset Numberbatch available from ConceptNet- https://github.com/commonsense/conceptnet-numberbatch. 
