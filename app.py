from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import re
import nltk
import spacy
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Download NLTK resources (only first time)
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('english'))

# -----------------------------
# Load Models
# -----------------------------
try:
    scaling_pipeline = joblib.load("models/scaling_pipeline.pkl")
    vectorization_pipeline = joblib.load("models/vectorization_pipeline.pkl")
    review_classifier_model = joblib.load("models/Review_classifier_LG.pkl")
    print("Models loaded successfully")
except Exception as e:
    print("Model loading error:", e)

# -----------------------------
# Load NLP tools
# -----------------------------
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# Text preprocessing
# -----------------------------
def preprocess(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]

    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]

    return " ".join(lemmas)

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = request.get_json()

        review_text = data.get("review_text", "")
        overall = int(data.get("overall", 5))
        helpful_ratio = float(data.get("helpful_ratio", 0))

        # -------- Feature Engineering --------

        words = review_text.split()

        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        review_length = len(review_text)

        # numeric features (5 features)
        numeric_features = np.array([[
            review_length,
            helpful_ratio,
            word_count,
            avg_word_length,
            overall
        ]])

        # SCALE ONLY NUMERIC FEATURES
        scaled_numeric = scaling_pipeline.transform(numeric_features)

        # -------- Text Processing --------

        cleaned = preprocess(review_text)

        vectorized = vectorization_pipeline.transform([cleaned])

        # -------- Combine Features --------

        from scipy.sparse import hstack

        final_features = hstack((scaled_numeric, vectorized))

        # -------- Prediction --------

        prediction = review_classifier_model.predict(final_features)[0]
        probability = review_classifier_model.predict_proba(final_features)[0]
        
        label = "AI-generated" if prediction == 1 else "Human-written"

        rating_map = {
            5: "Excellent Quality",
            4: "Very Good Quality",
            3: "Average Quality",
            2: "Poor Quality",
            1: "Very Poor Quality",
        }

        helpful_percent = round(helpful_ratio * 100)

        return jsonify({
            "prediction": label,
            "confidence": float(max(probability)),
            "rating_value": overall,
            "rating_description": rating_map.get(overall),
            "helpful_percentage": helpful_percent,
            "helpful_description": f"{helpful_percent}% users found this helpful"
        })

    except Exception as e:

        print("SERVER ERROR:", str(e))   # This will show real error in terminal

        return jsonify({
            "error": str(e)
        }), 500

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)