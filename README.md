# Stroke Risk Prediction (Logistic Regression + Pipeline)

End-to-end binary classification project to predict stroke risk using a public healthcare dataset.
This repository demonstrates a clean ML workflow with preprocessing + modeling inside a scikit-learn Pipeline, proper handling of class imbalance, and evaluation using ROC-AUC.

> Disclaimer: This project is for learning/demo purposes only. It is not medical advice and must not be used for clinical decision-making.

---

## What this project does
- Loads a healthcare stroke dataset (CSV)
- Performs quick sanity checks (shape, missing values, class distribution)
- Preprocesses data:
- Drops non-informative identifiers (e.g., `id`)
- Handles missing values (e.g., BMI)
- Encodes categorical features (one-hot / dummies)
- Scales numeric features using StandardScaler via ColumnTransformer
- Trains a Logistic Regression classifier with class_weight="balanced" to address class imbalance
- Evaluates performance using:
- Accuracy
- Confusion Matrix
- Classification Report (Precision/Recall/F1)
- ROC Curve + AUC
- Provides a simple new patient inference example that outputs:
- Predicted label (HIGH/LOW risk)
- Probability of stroke

---

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

'''
###Data set:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset


---

## How to run
### 1) Clone the repository
```bash
git clone https://github.com/<your-username>/stroke-risk-prediction.git
