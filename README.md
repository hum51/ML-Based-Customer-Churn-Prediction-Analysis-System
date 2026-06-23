# 🚀 ML-Based-Customer-Churn-Prediction-Analysis-System
ML Based Customer Churn Prediction &amp; Analysis System - A complete machine learning solution to predict customer churn. Uses XGBoost, Random Forest, Logistic Regression with 76% ROC-AUC. Includes EDA, feature engineering, SHAP analysis, and interactive prediction interface. Built with Python, Scikit-learn, Google Colab.

## 📋 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses, telecom companies, SaaS platforms, and e-commerce businesses. Losing customers directly impacts revenue and business growth.

This project builds a **Machine Learning system** that analyzes customer data, identifies patterns associated with customer churn, and predicts which customers are at risk of leaving.

### 🎯 Key Achievements
- ✅ **1,500+** Realistic Customer Records
- ✅ **5** Different ML Models Trained & Evaluated
- ✅ **76%** ROC-AUC Score Achieved
- ✅ **Interactive Prediction Interface** Built
- ✅ **SHAP Analysis** for Explainable AI
- ✅ **Business Recommendations** Provided

## 📊 Dataset Information

The dataset contains **1,500+ customer records** with the following attributes:

| Feature | Description |
|---------|-------------|
| 👤 CustomerID | Unique customer identifier |
| 📛 CustomerName | Customer's full name |
| 🎂 Age | Customer's age |
| 👫 Gender | Male / Female |
| 🏙️ City | Customer's city |
| 📦 SubscriptionType | Basic / Standard / Premium / Enterprise |
| 💰 MonthlySpending | Amount spent monthly |
| ⏱️ Tenure | Months with the company |
| 🛒 NumberOfPurchases | Total purchases made |
| 📞 CustomerSupportRequests | Number of support requests |
| 🔑 LoginFrequency | How often customer logs in |
| 📅 LastActivityDate | Last activity date |
| ⭐ SatisfactionScore | Customer satisfaction rating (1-5) |
| 🎯 ChurnStatus | Target variable (0 = No Churn, 1 = Churn) |

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Programming Language** | 🐍 Python 3.8+ |
| **Development Environment** | 📓 Google Colab |
| **Data Manipulation** | 📊 Pandas, NumPy |
| **Data Visualization** | 📈 Matplotlib, Seaborn |
| **Machine Learning** | 🤖 Scikit-learn, XGBoost |
| **Explainable AI** | 🔍 SHAP |
| **Model Saving** | 💾 Joblib |

## 🔧 Data Preparation & Feature Engineering

### Data Cleaning
- 🧹 Handled missing values in numerical columns with **median**
- 🧹 Handled missing values in categorical columns with **mode**
- 🧹 Removed duplicate records

### Feature Engineering (5 New Features Created)

| New Feature | Description |
|-------------|-------------|
| 📅 DaysSinceLastActivity | Days since customer's last activity |
| ⭐ EngagementScore | Combined metric of login frequency & purchases |
| 📊 SpendingToTenureRatio | Monthly spending divided by tenure |
| 🛒 AvgSpendingPerPurchase | Average amount spent per purchase |
| 📞 SupportRequestIntensity | Support requests per month of tenure |

### Data Encoding & Scaling
- 🔢 Label Encoding for binary categorical variables
- 🔢 One-Hot Encoding for multi-category variables
- 📏 StandardScaler for feature normalization

## 📈 Exploratory Data Analysis (EDA) - Key Insights

| Insight | Finding |
|---------|---------|
| 🎯 **Churn Rate** | ~25% of customers churned |
| ⭐ **Satisfaction Score** | Strongest predictor of churn |
| ⏱️ **Tenure** | Newer customers churn more |
| 📞 **Support Requests** | Higher requests = higher churn |
| 📦 **Subscription Type** | Premium users churn less |
| 💰 **Monthly Spending** | Higher spending = lower churn |

## 🤖 Machine Learning Models

### Models Trained

| # | Model | Description |
|---|-------|-------------|
| 1 | 🔷 Logistic Regression | Baseline linear model |
| 2 | 🌳 Decision Tree | Tree-based interpretable model |
| 3 | 🌲 Random Forest | Ensemble of decision trees |
| 4 | ⚡ Gradient Boosting | Sequential boosting algorithm |
| 5 | 🚀 XGBoost | Optimized gradient boosting |

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| 🚀 **XGBoost** | **0.73** | **0.72** | **0.71** | **0.71** | **0.76** |
| ⚡ Gradient Boosting | 0.72 | 0.71 | 0.70 | 0.70 | 0.75 |
| 🌲 Random Forest | 0.72 | 0.70 | 0.70 | 0.70 | 0.74 |
| 🌳 Decision Tree | 0.70 | 0.68 | 0.67 | 0.67 | 0.70 |
| 🔷 Logistic Regression | 0.69 | 0.67 | 0.66 | 0.66 | 0.68 |

## 🔍 Feature Importance Analysis

### Top Predictors of Churn

 ⭐ Satisfaction Score ████████████████████░░░░ │
│ ⏱️ Tenure ██████████████░░░░░░░░░░ │
│ ⭐ Engagement Score ████████████░░░░░░░░░░░░ │
│ 📦 SubscriptionType_Premium ████████░░░░░░░░░░░░░░░░ │
│ 📞 SupportRequests ██████░░░░░░░░░░░░░░░░░░ │
│ 💰 MonthlySpending █████░░░░░░░░░░░░░░░░░░░ │
│ 🔑 LoginFrequency ████░░░░░░░░░░░░░░░░░░░░ │
│ 🛒 NumberOfPurchases ███░░░░░░░░░░░░░░░░░░░░░ │
│ 👤 Age ██░░░░░░░░░░░░░░░░░░░░░░ │
│ 📅 DaysSinceLastActivity █░░░░░░░░░░░░░░░░░░░░░░ │

### Business Impact
- 📌 **Satisfaction Score** is the #1 predictor - Focus on improving customer satisfaction
- 📌 **Tenure** is critical - Retain new customers in first 6 months
- 📌 **Engagement Score** matters - Keep customers engaged

## 🎯 Prediction Interface

### Features
- 📝 Input customer details manually
- ⚡ Instant churn probability calculation
- 🏷️ Churn status prediction (Yes/No)
- 📊 Risk level assessment (High/Medium/Low)

### Example Prediction

```python
# Sample Customer Input
customer = {
    'Age': 35,
    'Gender': 'Male',
    'City': 'Karachi',
    'SubscriptionType': 'Basic',
    'MonthlySpending': 500.0,
    'Tenure': 12,
    'NumberOfPurchases': 15,
    'CustomerSupportRequests': 2,
    'LoginFrequency': 18,
    'SatisfactionScore': 3.5
}

# Output
{
    'churn_probability': 0.68,
    'churn_prediction': 1,
    'churn_status': 'Yes',
    'risk_level': 'Medium'
}

🏢 Business Recommendations
#	Recommendation	Impact
1	⭐ Focus on Customer Satisfaction - Regularly measure and improve satisfaction scores	High
2	🆕 Early Intervention for New Customers - Focus on first 6 months of tenure	High
3	📞 Proactive Support - Monitor support request patterns and reach out early	Medium
4	📦 Encourage Premium Upgrades - Offer incentives to upgrade from Basic plans	Medium
5	💰 Loyalty Programs - Keep customers engaged with rewards	Medium
6	🚨 Early Warning System - Identify at-risk customers before they leave	High
7	🔄 Regular Model Updates - Retrain models with new data monthly	Low

Expected Outcomes
📉 15-25% Reduction in customer churn
💵 20-30% Increase in customer lifetime value
📈 10-15% Improvement in customer satisfaction scores

🔮 Future Improvements
🌐 Deploy as Web Application using Flask/Streamlit
📊 Real-time Churn Monitoring Dashboard
🧠 Deep Learning with Neural Networks
📈 A/B Testing Integration
🏷️ Customer Segmentation Analysis
