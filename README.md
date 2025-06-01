# Credit Card Fraud Detection

**Author:** Sonia James

---

## Executive Summary

### Rationale

Credit card fraud is a serious and costly problem affecting consumers, businesses, and financial institutions worldwide. Detecting fraudulent transactions early helps prevent significant financial losses, protects customers’ sensitive information, and maintains trust in digital payment systems.

With the increasing volume of online and contactless transactions, fraudsters constantly evolve their tactics, making it critical to have effective, data-driven methods to identify suspicious activity quickly and accurately. Improved fraud detection safeguards both companies and consumers from the negative impacts of fraud, such as monetary loss, legal liabilities, and damaged reputations.

Therefore, studying and improving credit card fraud detection models is essential to enhancing security, reducing fraud-related costs, and ensuring safer financial transactions for everyone.

---

## Research Question

**Can we accurately detect whether a given transaction is fraudulent or not?**

---

## Data Sources

The dataset used for this project is sourced from Kaggle:  
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/dntai1983/fraud-data/data)

---

## Methodology

To address the problem of credit card fraud detection, the following methods were used:

- **Data Cleaning and Preparation:**
  Handling missing values, encoding categorical variables (using target encoding for high-cardinality features and one-hot encoding for low-cardinality features), and scaling numerical features to prepare the dataset for modeling.

- **Exploratory Data Analysis (EDA):**  
  Visualizing key features such as transaction amount, age, time of transaction, and location to understand patterns related to fraudulent behavior.

- **Feature Engineering:**  
  Extracting meaningful features from transaction timestamps (hour, day, weekend flag) and incorporating location data to help the model detect anomalies.

## Modeling

We tested the following classification models:

- **Random Forest Classifier**
- **Logistic Regression**
- **XGBoost Classifier**

All models handled class imbalance using `class_weight='balanced'` or equivalent techniques.

---

## Model Evaluation

Models were evaluated using:

- **ROC AUC Score** – Area under the ROC curve  
- **F1 Score** – Harmonic mean of precision and recall  
- **Precision & Recall** – Accuracy of fraud predictions  
- **Confusion Matrix** – TP, FP, TN, FN breakdown

---

###  Model Comparison

| Model                | ROC AUC | F1 Score | Precision | Recall | Accuracy |
|----------------------|---------|----------|-----------|--------|----------|
|   Random Forest      | **0.9945** | **0.8414** | **0.9537** | **0.7528** | **0.9985** |
|   Logistic Regression| 0.9334  | 0.0841   | 0.0445    | 0.7753 | 0.9115   |
|   XGBoost            | 0.8796  | 0.0447   | 0.1915    | 0.0253 | 0.9943   |


---

## Results

Key findings from the data analysis include:

- High-value transactions by younger users are more likely to be fraudulent.
- Larger transaction amounts tend to have a slightly higher risk of fraud.
- Fraudulent transactions increase during holiday seasons.
- Fraudulent activities are more common during nighttime hours.
  <img width="409" alt="Screenshot 2025-06-01 at 1 37 12 PM" src="https://github.com/user-attachments/assets/0683b9c3-cfa3-4986-982c-bc444c63065c" />


Model performance :

- **Random Forest** significantly outperformed other models across all evaluation metrics.
- **Logistic Regression** and **XGBoost** underperformed due to:
  - Class imbalance
  - Inability to capture complex non-linear relationships
    
  <img width="1544" alt="Screenshot 2025-06-01 at 1 29 47 PM" src="https://github.com/user-attachments/assets/ecaacc37-a195-408b-9f63-53640d2ff767" />


---  
## Outline of project
https://github.com/sonia-james/AIML_Capstone_Final/blob/main/Capstore_Fraud_Detection_Final.ipynb

## Contact and Further Information
Sonia James 

LinkedIn : https://www.linkedin.com/in/sonia-james-6b1595a9/
