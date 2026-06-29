import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===========================
# Load Datasets
# ===========================

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# ===========================
# Add Labels
# Fake = 0
# Real = 1
# ===========================

fake["label"] = 0
real["label"] = 1

# ===========================
# Combine Datasets
# ===========================

data = pd.concat([fake, real], ignore_index=True)

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ===========================
# Combine Title + Text
# ===========================

data["content"] = (
    data["title"].fillna("") + " " + data["text"].fillna("")
)

# Keep required columns
data = data[["content", "label"]]

# ===========================
# TF-IDF Vectorization
# ===========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=50000
)

X = vectorizer.fit_transform(data["content"])
y = data["label"]

print("Text converted successfully!")
print("Feature Matrix Shape:", X.shape)

# ===========================
# Split Dataset
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ===========================
# Train Model
# ===========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully!")

# ===========================
# Accuracy
# ===========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# ===========================
# Save Model
# ===========================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and Vectorizer saved successfully!")