# Spam-Ham Classification using NLP

## 📌 Project Overview
This project focuses on **Spam vs. Ham (Non-Spam) classification** using **Natural Language Processing (NLP)** and machine learning techniques. The goal is to develop a classifier that accurately distinguishes spam messages from legitimate messages.

## 📂 Dataset
- **Dataset Name:** `SMSSpamCollection`
- **Source:** [Publicly available SMS spam dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Contents:**
  - Text messages (SMS)
  - Labels (`spam` or `ham`)

## 🔧 Preprocessing Steps
1. **Text Cleaning:** Removing special characters, numbers, and punctuation.
2. **Tokenization:** Splitting messages into individual words.
3. **Stopword Removal:** Eliminating common words that do not add meaning.
4. **Stemming/Lemmatization:** Reducing words to their root form.
5. **Vectorization:** Converting text data into numerical representations using **TF-IDF** or **CountVectorizer**.

## 🏗️ Machine Learning Models Used
- **Baseline Models:**
  - Logistic Regression
  - Naive Bayes
- **Ensemble Learning Models:**
  - Random Forest Classifier (**Primary Focus**)
  - Gradient Boosting Models (GBM)
- **Evaluation Metrics:**
  - Accuracy
  - Precision, Recall, and F1-score
  - Confusion Matrix

## 📊 Model Comparison
| Model | False Positives (FP) | False Negatives (FN) | Key Concern |
|--------|----------------------|----------------------|-------------|
| **Random Forest 1 (RF1)** | 21 | 15 | Higher false negatives (spam misclassified as ham) |
| **Random Forest 2 (RF2)** | 14 | 2 | Lower false negatives, making it a better choice |

## 🏆 Why RF2 is the Best Model?
- **Minimizes False Negatives (FN)**: Only 2 spam messages misclassified as ham, reducing the risk of spam going undetected.
- **Lower False Positives (FP)**: Slightly fewer false positives compared to RF1, meaning fewer legitimate messages are flagged as spam.
- **Balanced Performance**: Achieves a good trade-off between precision and recall.

## 🚀 Next Steps
- **Fine-tune hyperparameters** for further improvements.
- **Deploy the model** as an API or web app for real-world spam detection.
- **Experiment with deep learning** models such as LSTMs or Transformers.

## 📝 Repository Structure
```
|-- data/
|   |-- SMSSpamCollection  # Raw dataset
|-- notebooks/
|   |-- NLP Project-Spam Ham Classification.ipynb  # Full code and analysis
|-- README.md  # This file
```

## 👨‍💻 Author
Akanksha Gayaprasad Kushwaha
