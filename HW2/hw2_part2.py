# -*- coding: utf-8 -*-
"""HW2_Part2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uiDIF6UQZpaIQzQHr3dCBIvqrGDfV3Md
"""

# NOTE: As of 2/22/2022, Dr. Liu allowed the use of sklearn preprocessing methods along with
# spaCy to develop the classifier.  She also allowed the use of sklearn's premade 20newsgroups dataset.

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import re

"""# Model 1: Pure Sklearn Setup

Sklearn and re library documentation was used in the creation of this program. 
"""

# Extract only the train and test datasets for our categories and remove unecessary components
categories = ['rec.autos', 'comp.graphics']
train = datasets.fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True)
test = datasets.fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

# Get the number of documents in each category for train set
cat1 = 0
cat2 = 0
for i in range(len(train.target)):
  if train.target[i] == 0:
    cat1 += 1
  else:
    cat2 += 1

# Remove all numbers and special characters from document texts
bad_patterns = "[^a-zA-Z. ]"

for doc in range(len(train.data)):
  new_doc = re.sub(bad_patterns, '', train.data[doc])
  train.data[doc] = new_doc

# Tokenize all documents in our training set and get the vocabulary.
vectorizer = CountVectorizer()
vectorizer.fit_transform(train.data)
vocabulary = vectorizer.vocabulary_
print(len(vocabulary))

# Use TFidfVectorizer() as initial pipeline to handle current setup of the dataset.
model = pipeline.make_pipeline(CountVectorizer(), TfidfTransformer(), MultinomialNB())

model.fit(train.data, train.target)

predicted = model.predict(test.data)

print("Number of documents in rec.autos: " + str(cat1))
print("Number of documents in comp.graphics: " + str(cat2))
print("Vocabulary Size: " + str(len(vocabulary)))
print(metrics.classification_report(test.target, predicted, target_names=test.target_names))

"""# Model 2: Combo of Sklearn and SpaCy Features

Predefined sklearn pipeline is used below and then fed
to sklearn's CountVectorizer method.

Sklearn, re, and spaCy documentation was used in the creation of this program.

"""

import spacy as sp
from spacy.lang.en.stop_words import STOP_WORDS

# Re-import unedited datasets for new model
train2 =  datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories,  shuffle=True)
test2 = datasets.fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

# Remove unecessary characters like numbers and special non-punctuation characters
bad_patterns = "[^a-zA-Z.]"

for doc in range(len(train2.data)):
  new_doc = re.sub(bad_patterns, '', train2.data[doc])
  train2.data[doc] = new_doc

# Process each document individually using the below steps
# NOTE: Colab's spaCy library version is 2.2.4.  Only version 3.0 has 
# lemmatization as a separate pipeline component.  Thus, lemmatization is
# implemented, but acts behind the scenes of the parser component.

# Import premade English processing pipeline
nlp = sp.load("en_core_web_sm")

# Add sentence segmentation
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

# spaCy pipeline for pre-processing
def spacy_pipeline(document):
  doc = nlp(document)
  return doc

# Create bag of words with spacy tokenizer
vectorizer = CountVectorizer(tokenizer=spacy_pipeline)

# Train the model
# model2 = pipeline.Pipeline([("bow", bag_of_words), ("classifier", classifier)])
model2 = pipeline.make_pipeline(vectorizer, TfidfTransformer(), MultinomialNB())
model2.fit(train2.data, train2.target)

predicted2 = model2.predict(test2.data)

print(metrics.classification_report(test2.target, predicted2, target_names=test2.target_names, zero_division=1))