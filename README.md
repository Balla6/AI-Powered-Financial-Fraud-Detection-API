# AI-Powered Financial Fraud Detection API

## 📌 Overview

The **AI-Powered Financial Fraud Detection API** is a real-time fraud detection system designed to identify fraudulent transactions using machine learning. This project leverages **SQL, Python, XGBoost, and Google BigQuery** to process transactions, detect fraud patterns, and provide actionable insights for businesses in **finance, banking, and e-commerce**.

🚀 **Live API Endpoint:** [https://fraud-detection-api-ybnj.onrender.com](https://fraud-detection-api-ybnj.onrender.com)

---

## 🎯 **Project Goals**
- ✅ Detect fraudulent transactions using AI models.
- ✅ Provide real-time fraud risk scores for each transaction.
- ✅ Make fraud detection scalable & marketable for businesses.

---

## 📂 **Dataset Used**
- **Dataset:** Credit Card Fraud Detection Dataset (**284,807 transactions**)
- **Final Processed Shape:** `(8,580,255, 26)`
- **Stored in:** Google BigQuery (Processed in Chunked Parquet Format)

---

## 🛠️ **Tools & Technologies**
✔ **SQL (Google BigQuery):** Data storage & feature engineering  
✔ **Python:** Machine learning (XGBoost), data analysis (Pandas, NumPy)  
✔ **Google BigQuery:** Feature engineering on large datasets  
✔ **FastAPI + Render:** API Deployment  

---

## 📊 **Exploratory Data Analysis (EDA)**
Performed detailed EDA on:
- 🔹 **Fraud vs. Non-Fraud Distribution**
- 🔹 **Transaction Amount Distribution**
- 🔹 **Fraud Rate per Transaction Category**
- 🔹 **Customer Behavior Analysis**
- 🔹 **Merchant Behavior Analysis**

### 🔍 **Cleaned Data Columns:**
```
['trans_num', 'trans_date', 'trans_time', 'unix_time', 'category', 'amt',
 'is_fraud', 'merchant', 'merch_lat', 'merch_long',
 'customer_num_trans_1_day', 'customer_num_trans_7_day',
 'customer_num_trans_30_day', 'trans_time_is_night', 'trans_time_day',
 'customer_avg_amount_1_day', 'customer_avg_amount_7_day',
 'customer_avg_amount_30_day', 'merchant_num_trans_1_day',
 'merchant_num_trans_7_day', 'merchant_num_trans_30_day',
 'merchant_risk_1_day', 'merchant_risk_7_day', 'merchant_risk_30_day',
 'merchant_risk_90_day', 'transaction_hour']
```

---

## 🚀 **Feature Engineering**
**Enhanced Features using Google BigQuery:**
```sql
CREATE OR REPLACE TABLE fraud_detection_project.transactions_enhanced AS
SELECT *,
    COUNT(CASE WHEN is_fraud = 1 THEN 1 END) OVER (PARTITION BY trans_num) / COUNT(*) OVER (PARTITION BY trans_num) AS customer_risk_score,
    AVG(amt) OVER (PARTITION BY trans_num) AS customer_avg_transaction_amount,
    COUNT(CASE WHEN is_fraud = 1 THEN 1 END) OVER (PARTITION BY merchant) / COUNT(*) OVER (PARTITION BY merchant) AS merchant_fraud_rate,
    CASE WHEN COUNT(CASE WHEN is_fraud = 1 THEN 1 END) OVER (PARTITION BY merchant) / COUNT(*) OVER (PARTITION BY merchant) > 0.1 THEN 1 ELSE 0 END AS merchant_high_risk_flag
FROM fraud_detection_project.transactions;
```
✅ **Loaded Processed Data from BigQuery into Pandas**  
✅ **Performed SMOTE to Balance Dataset**  
✅ **Applied Feature Scaling & Normalization**  

---

## 📈 **Model Training & Performance**

### 🚀 **Initial XGBoost Model:**
- **Accuracy:** `1.0` (Overfitting detected, so improvements were made)
- **AUC-ROC Score:** `1.0`

### ✅ **Improved Model (XGBoost with Hyperparameter Tuning):**
```python
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
```
- **Updated Accuracy:** `96.50%`
- **AUC-ROC Score:** `0.9650`

### **Final Classification Report:**
| Class             | Precision | Recall | F1-Score |
| ----------------- | --------- | ------ | -------- |
| **Fraud (1)**     | 0.98      | 0.95   | 0.96     |
| **Non-Fraud (0)** | 0.95      | 0.98   | 0.97     |

---

## 🚀 **API Deployment (FastAPI + Render)**
### ✅ **API Features:**
- **POST** `/predict` → Returns fraud probability for a given transaction.
- **GET** `/docs` → FastAPI Swagger UI for testing.

### 📌 **How to Use the API**
#### 🔹 **Python Example:**
```python
import requests

url = "https://fraud-detection-api-ybnj.onrender.com/predict"
data = {
    "trans_date": "2025-03-07",
    "unix_time": 1678923000,
    "amt": 1200.50,
    "merch_lat": 40.7128,
    "merch_long": -74.0060,
    "customer_num_trans_1_day": 5,
    "customer_num_trans_7_day": 20,
    "customer_num_trans_30_day": 50,
    "trans_time_is_night": 0,
    "trans_time_day": 1,
    "trans_date_is_weekend": 0,
    "customer_avg_amount_1_day": 500.00,
    "customer_avg_amount_7_day": 400.00,
    "customer_avg_amount_30_day": 350.00,
    "merchant_num_trans_1_day": 100,
    "merchant_num_trans_7_day": 700,
    "merchant_num_trans_30_day": 3000,
    "merchant_risk_1_day": 0.02,
    "merchant_risk_7_day": 0.05,
    "merchant_risk_30_day": 0.07,
    "merchant_risk_90_day": 0.1,
    "transaction_hour": 14,
    "customer_avg_transaction_amount": 420.00,
    "merchant_fraud_rate": 0.02,
    "merchant_high_risk_flag": 0,
    "hourly_fraud_rate": 0.03,
    "is_high_risk_hour": 0
}

response = requests.post(url, json=data)
print(response.json())  # {'fraud_probability': 0.82}
```

---

## 🔮 **Future Improvements & Business Potential**
✅ **Sell the system as a fraud detection service to fintech companies**  
✅ **Add real-time fraud alerts**  
✅ **Deploy a full SaaS dashboard with Power BI**  
✅ **Implement a fraud reporting automation pipeline using Apache Airflow** *(planned for future use)*  

---

## 📝 **License & Credits**
- 📌 **Dataset:** Credit Card Fraud Detection Dataset
- 📌 **License:** MIT License (or specify another if needed)
- 📌 **Author:** [Balla Diaite](https://github.com/Balla6)

---

### 🚀 **Final Notes**
🎉 **This project is now fully live and production-ready!**

💡 If you’d like to contribute, **fork the repo and submit a PR!** 🔥


