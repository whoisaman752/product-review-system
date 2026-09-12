import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# =========================
# Load Dataset
# =========================

df = pd.read_csv("review_dataset.csv")

# Rename columns for clarity
df = df.rename(columns={
    "text": "review",
    "generated": "label"
})

# =========================
# Feature Engineering
# =========================

# review length
df["reviewLength"] = df["review"].apply(len)

# word count
df["wordCount"] = df["review"].apply(lambda x: len(x.split()))

# average word length
df["avgWordLength"] = df["review"].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
)

# Since dataset doesn't contain these,
# we generate synthetic values to match app.py input

np.random.seed(42)

df["helpful_ratio"] = np.random.uniform(0, 1, len(df))
df["overall"] = np.random.randint(1, 6, len(df))

# =========================
# Define Features
# =========================

text_data = df["review"]

numeric_features = df[[
    "reviewLength",
    "helpful_ratio",
    "wordCount",
    "avgWordLength",
    "overall"
]]

labels = df["label"]

# =========================
# TF-IDF Vectorization
# =========================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_text = vectorizer.fit_transform(text_data)

# =========================
# Scaling Numeric Features
# =========================

scaler = StandardScaler()

X_numeric = scaler.fit_transform(numeric_features)

# =========================
# Combine Features
# =========================

X_final = hstack((X_numeric, X_text))

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    labels,
    test_size=0.2,
    random_state=42
)

# =========================
# Train Model
# =========================

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

# =========================
# Save Models
# =========================

joblib.dump(vectorizer, "models/vectorization_pipeline.pkl")
joblib.dump(scaler, "models/scaling_pipeline.pkl")
joblib.dump(model, "models/Review_classifier_LG.pkl")

print("✅ Model training completed")
print("✅ Files saved in models folder")

