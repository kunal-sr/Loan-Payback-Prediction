# 💰 Predicting Loan Payback

This project focuses on predicting whether a loan will be paid back or not using machine learning techniques. The solution is built using LightGBM and evaluated using the ROC-AUC score, making it suitable for binary classification problems with imbalanced data.

## 📌 Overview

Financial institutions face risks when issuing loans. This project aims to build a predictive model that determines the likelihood of a borrower repaying a loan, helping reduce financial risk and improve decision-making.

## 📂 Dataset

The dataset consists of:

- train.csv → Training data with features and target (loan_paid_back)

- test.csv → Test data without target labels

- sample_submission.csv → Format for submission

- submission.csv → Final predictions generated

## ⚙️ Approach

1. Data Loading

- Loaded datasets using Pandas

- Checked dataset shape and previewed data

2. Data Preprocessing

- Removed unnecessary columns (id)

- Handled missing values by replacing them with -999

- Encoded categorical variables using Label Encoding

3. Train-Test Split

- Split training data into:

  80% training

  20% validation

- Used stratified sampling to maintain class balance

4. Model Used

- LightGBM Classifier

- Fast and efficient gradient boosting model

- Handles large datasets and missing values well

- Model Parameters:

```
LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```
5. Evaluation Metric

- Used ROC-AUC Score to evaluate model performance

- Suitable for binary classification problems

6. Final Training

- Model retrained on full dataset after validation

7. Prediction

- Generated probability predictions for test data

- Saved results in submission.csv

## 📊 Results

- Validation Metric: ROC-AUC Score

- Model shows strong performance in distinguishing between loan repayment outcomes

## 🛠️ Tech Stack

- Python 🐍

- Pandas

- Scikit-learn

- LightGBM

## 🚀 How to Run

Clone the repository:
```
git clone https://github.com/your-username/loan-payback-prediction.git
```
```
cd loan-payback-prediction
```

Install dependencies:

```
pip install pandas scikit-learn lightgbm
```

Run the script:

```
python test.py
```

Output:

submission.csv will be generated

## 📁 Project Structure

```
├── train.csv
├── test.csv
├── sample_submission.csv
├── submission.csv
├── test.py
└── README.md
```

## 💡 Key Insights

- Handling missing values effectively is crucial

- Label encoding works well for tree-based models

- LightGBM provides high performance with minimal tuning

- Stratified splitting ensures better validation reliability

## 🔮 Future Improvements

- Hyperparameter tuning (Optuna/GridSearch)

- Feature engineering

- Cross-validation instead of single split

- Handling categorical variables using advanced encoding (e.g., Target Encoding)
