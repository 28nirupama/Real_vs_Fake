# train_model.py
import pandas as pd
import numpy as np
import joblib
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from scipy.sparse import hstack

from extra_features import ExtraFeatures  # import the shared module

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.strip()

# ----- Load data -----
df = pd.read_csv("RF_data.csv")  # ensure file is in same folder
print("Raw CSV shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check expected columns
expected_cols = {"Human_Content", "AI_Content"}
if not expected_cols.issubset(set(df.columns)):
    raise ValueError(f"CSV is missing expected columns. Found: {df.columns.tolist()}")

# ----- Build dataframes correctly -----
human_df = pd.DataFrame({"text": df["Human_Content"], "label": "human"})
ai_df = pd.DataFrame({"text": df["AI_Content"], "label": "ai"})

df_final = pd.concat([human_df, ai_df], ignore_index=True)
df_final = df_final.dropna(subset=["text"]).reset_index(drop=True)
df_final["clean_text"] = df_final["text"].apply(clean_text)

print("After cleaning:", df_final.shape)
print(df_final["label"].value_counts())

# ----- Basic sanity checks -----
if df_final.shape[0] == 0:
    raise ValueError("No rows left after cleaning. Check your CSV and preprocessing.")

# ----- Balance (oversample minority) -----
human = df_final[df_final.label == "human"]
ai = df_final[df_final.label == "ai"]

if len(human) == 0 or len(ai) == 0:
    raise ValueError("One of the classes has zero samples. Check your CSV content.")

if len(human) > len(ai):
    ai = resample(ai, replace=True, n_samples=len(human), random_state=42)
elif len(ai) > len(human):
    human = resample(human, replace=True, n_samples=len(ai), random_state=42)

df_balanced = pd.concat([human, ai]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Balanced shape:", df_balanced.shape)
print(df_balanced['label'].value_counts())

# Safety: ensure at least 2 samples per class for stratify
min_class_count = df_balanced['label'].value_counts().min()
if min_class_count < 2:
    raise ValueError("Need at least 2 samples per class after balancing to proceed with stratified split.")

# ----- Split -----
X = df_balanced["clean_text"]
y = df_balanced["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
print("Train/Test sizes:", X_train.shape, X_test.shape)

# ----- Vectorize -----
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,3), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ----- Extra features -----
extra = ExtraFeatures()
X_train_extra = extra.fit_transform(X_train)
X_test_extra = extra.transform(X_test)

# combine
X_train_combined = hstack([X_train_tfidf, X_train_extra])
X_test_combined = hstack([X_test_tfidf, X_test_extra])

# ----- Train model -----
model = LinearSVC()
model.fit(X_train_combined, y_train)

# ----- Evaluate -----
y_pred = model.predict(X_test_combined)
print("\nMODEL REPORT:\n")
print(classification_report(y_test, y_pred))

# ----- Save model and vectorizer (we'll recreate ExtraFeatures in app) -----
joblib.dump(model, "ai_human_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("\nModel and vectorizer saved: ai_human_model.pkl, vectorizer.pkl")
