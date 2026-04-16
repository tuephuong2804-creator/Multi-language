import os
import pandas as pd
import zipfile
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from transformers import pipeline

# DOWNLOAD DATASET
if not os.path.exists("Language Detection.csv"):
    print("Downloading dataset...")
    os.system("kaggle datasets download -d basilb2s/language-detection")

    print("Extracting dataset...")
    with zipfile.ZipFile("language-detection.zip", 'r') as zip_ref:
        zip_ref.extractall()

# LOAD DATA
data = pd.read_csv("Language Detection.csv")

print("\n=== DATASET INFO ===")
print("Total samples:", len(data))
print("Shape:", data.shape)

print("\nSamples per language:")
print(data["Language"].value_counts())

# FILTER LANGUAGES
languages = ["English", "French", "Spanish", "Italian"]
data = data[data["Language"].isin(languages)]

print("\nFiltered data distribution:")
print(data["Language"].value_counts())

# VISUALIZE LANGUAGE DISTRIBUTION
data["Language"].value_counts().plot(kind="bar")
plt.title("Language Distribution")
plt.show()

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    data["Text"],
    data["Language"],
    test_size=0.2,
    stratify=data["Language"],
    random_state=42)

# FEATURE EXTRACTION (TF-IDF)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# MODEL 1: NAIVE BAYES
model_nb = MultinomialNB()
model_nb.fit(X_train_vec, y_train)

y_pred_nb = model_nb.predict(X_test_vec)

print("\n=== NAIVE BAYES RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_nb))

# Confusion Matrix NB
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=model_nb.classes_)
disp_nb.plot(cmap="Greens")
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# MODEL 2: TRANSFORMER
print("\n=== TRANSFORMER (BERT) RESULTS ===")

classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection")

# Mapping labels 
mapping = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian"}

labels = ["English", "French", "Spanish", "Italian"]

y_pred_bert = []

for text in X_test:
    result = classifier(text[:100])  
    label = result[0]["label"]
    label = mapping.get(label, label)
    y_pred_bert.append(label)

# Evaluate Transformer
print("Accuracy:", accuracy_score(y_test, y_pred_bert))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_bert))

# Confusion Matrix Transformer
cm_bert = confusion_matrix(y_test, y_pred_bert, labels=labels)
disp_bert = ConfusionMatrixDisplay(confusion_matrix=cm_bert, display_labels=labels)
disp_bert.plot(cmap="Blues")
plt.title("Transformer Confusion Matrix")
plt.show()
<<<<<<< HEAD
=======

>>>>>>> e18e1a5 (update README and main code)
