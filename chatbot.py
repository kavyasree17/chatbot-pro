import json
import random
import pickle
import numpy as np
import re
from nltk.stem import WordNetLemmatizer

# Load saved model, vectorizer, and classes
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

# Preprocess input (must match training!)
def preprocess(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

# Predict tag and return response
def chatbot_response(message):
    processed = preprocess(message)
    X = vectorizer.transform([processed])
    prediction = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])
    
    return "I didn't quite get that. Can you try saying it differently?"
