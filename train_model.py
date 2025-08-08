import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json", encoding="utf-8") as file:
      data = json.load(file)

corpus = []
labels = []
classes = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Preprocess
corpus = [lemmatizer.lemmatize(word.lower()) for word in corpus]

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = labels

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

print("✅ Model trained and saved.")
