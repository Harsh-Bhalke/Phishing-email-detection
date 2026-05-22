# Phishing Email Detection Model using Scikit-learn

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
# Dataset should contain:
# text  -> email content
# label -> phishing or safe

df = pd.read_csv("emails.csv")

# Convert labels to numeric
df['label'] = df['label'].map({'safe': 0, 'phishing': 1})

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess(text):
    text = str(text).lower()

    # Replace URLs with keyword
    text = re.sub(r'http[s]?://\S+', ' URL ', text)

    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)

    return text

# Clean email text
df['clean_text'] = df['text'].apply(preprocess)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Build Machine Learning Pipeline
# -----------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2)
    )),
    ('clf', LogisticRegression())
])

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix Visualization
# -----------------------------
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center', color='black')

plt.colorbar()
plt.show()

# -----------------------------
# Test Custom Email
# -----------------------------
sample_email = [
    "Congratulations! Your bank account is suspended. Click http://fake-link.com to verify now."
]

sample_email = [preprocess(email) for email in sample_email]

prediction = model.predict(sample_email)

print("\nSample Email Prediction:")
print("Phishing" if prediction[0] == 1 else "Safe")
