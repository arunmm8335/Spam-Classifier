from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load trained model
model_path = "models/spam_classifier.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Run spam_classifier.py first.")

with open(model_path, "rb") as f:
    vectorizer, model = pickle.load(f)
