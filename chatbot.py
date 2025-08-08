import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Use local nltk_data folder
nltk.data.path.append('./nltk_data')

lemmatizer = WordNetLemmatizer()

# Load saved files
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)


def preprocess_input(text):
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

def predict_intent(user_input):
    processed = preprocess_input(user_input)
    X = vectorizer.transform([processed])
    probs = model.predict_proba(X)[0]
    max_prob = max(probs)
    pred_class = model.predict(X)[0]
    if max_prob > 0.3:
        return pred_class
    else:
        return None

def get_response(intent):
    for item in intents['intents']:
        if item['tag'] == intent:
            return random.choice(item['responses'])
    return "I'm here to listen. Tell me more."

def chatbot_response(user_input):
    intent = predict_intent(user_input)
    if intent:
        return get_response(intent)
    else:
        return "I didn't quite get that. Can you try saying it differently?"
