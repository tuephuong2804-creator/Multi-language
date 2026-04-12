🌍 Multilingual Language Detection
📌 Overview

This project focuses on language detection across multilingual texts using both traditional machine learning and transformer-based approaches.

The goal is to compare:

A baseline model: Multinomial Naive Bayes (NB)
A transformer model: XLM-RoBERTa

The study also analyzes how models handle linguistically similar languages, especially within the Romance language family.

🎯 Objectives
Implement and evaluate two different models for language detection
Compare performance using:
Accuracy
Precision
Recall
F1-score
Analyze classification errors from a linguistic perspective
📊 Dataset
Source: Kaggle (Language Detection dataset)
Total samples: ~10,000
Filtered languages:
English
French
Spanish
Italian

These languages were selected to examine:

Differences between language families
Similarities within Romance languages
⚙️ Methodology
🔹 Data Processing
Load dataset from CSV
Filter selected languages
Train-test split (80/20, stratified)
🔹 Model 1: Naive Bayes
Feature extraction: TF-IDF (character n-grams: 2–4)
Classifier: Multinomial Naive Bayes
🔹 Model 2: Transformer
Model: XLM-RoBERTa (pre-trained)
Direct text classification using HuggingFace pipeline
📈 Results
Model	Accuracy
Naive Bayes	0.9898
Transformer (XLM-R)	0.9796
Key Findings:
Naive Bayes performs slightly better in overall accuracy
Transformer performs better on closely related languages
Errors mainly occur among Romance languages
🔍 Error Analysis
Confusion occurs between:
French, Spanish, Italian
Causes:
Shared vocabulary (e.g., information, nation)
Similar morphology and spelling
Transformer reduces confusion with unrelated languages
🛠️ Technologies Used
Python
pandas
scikit-learn
matplotlib
transformers (HuggingFace)
PyTorch
🚀 How to Run
1. Clone repository
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Install dependencies
pip install -r requirements.txt
3. Run the project
python main.py
