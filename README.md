# Predicting Star Rating of Reviews on Yelp based on Sentiment Analysis of Annotated Reviews Data Set

## Introduction
Yelp reviews consist of a star rating (lowest number indicates poor and highest number indicates excellent) as well as text reviews. In this project, we will be predicting the star rating corresponding to a text review. 

## Preprocessing Data
The data has to be cleaned and preprocessed first before feeding to our training model. NLTK's stopwords are used to eliminate the most commonly used words. Only reviews in English are considered. The selected reviews are converted to lowercase and special characters are removed.

Command to run the preProcess file. It takes two arguments as its input - business.json and review.json:

```python
python3 preProcess.py business.json review.json 
```

Expected Output will be a processed csv file with name "processedFile.csv"

```
processedFile.csv
```

## Model 

GRU is used for training the model with AdamOptimizer. Word embeddings for the model are taken from the open dataset Numberbatch available from ConceptNet- https://github.com/commonsense/conceptnet-numberbatch. We have a common file for both training and testing the model. It takes our processedFile.csv and an argument describing the mode (train or test).

### Training Model
Command to run the model file for training. Argument passed is "train".

```python
python3 model.py train
```
The model will be saved inside the same (current) folder. 

### Testing Model
Command to run the model file for testing. Argument passed is "test".

```python
python3 model.py test
```

## Output

Expected Output will be the accuracy printed out onto the console and confusion matrix will also be displayed.

> Test accuracy is: 0.5645978009259259
