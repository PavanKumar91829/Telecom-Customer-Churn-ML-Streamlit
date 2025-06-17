# Telecom Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn in a telecom company using machine learning. Customer churn, or the rate at which customers stop doing business with a company, is a critical metric for telecom companies. By identifying customers who are likely to churn, companies can take proactive measures to retain them.

## Problem Statement
The goal is to predict whether a telecom customer will churn (leave the service) based on their demographic information, service usage patterns, and billing details.

## Dataset
The project uses the Telco Customer Churn dataset, which includes information about:
- Customer demographics (gender, age, partners, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, online security, streaming TV, etc.)
- Billing information (monthly charges, total charges)
- Churn status (whether the customer left the company)

## Project Structure
```
├── telecom_customer_churn_logistic_regression.py  # Main analysis script
├── Streamlit/
│   └── Telecom_customer_churn_LogisticRegression.py  # Streamlit web application
├── Telco-Customer-Churn.csv  # Dataset file
└── README.md
```

## Features
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of customer data to identify patterns and relationships
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features
- **Machine Learning Model**: Logistic Regression model to predict customer churn
- **Interactive Web Application**: Streamlit-based interface for real-time churn prediction

## Key Insights from EDA
- Only 16.2% of customers are senior citizens
- Tenure ranges from 0 to 72 months
- Monthly charges distribution is slightly left-skewed (more customers paying less)
- Total charges distribution is right-skewed
- Most customers who churn leave within their first year
- Contract type, internet service, online security, and tech support are strong indicators of churn

## Model Performance
The Logistic Regression model achieves good accuracy in predicting customer churn. The exact metrics are displayed in the Streamlit application.

## How to Run the Project

### Prerequisites
- Python 3.7+
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit

### Installation
```bash
# Clone the repository (if applicable)
git clone <repository-url>

# Navigate to the project directory
cd "Project - 3 (Telecom Customer Churn)"

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### Running the Analysis Script
```bash
python telecom_customer_churn_logistic_regression.py
```

### Running the Streamlit App
```bash
cd Streamlit
streamlit run Telecom_customer_churn_LogisticRegression.py
```

## Using the Streamlit Application
1. The application displays the model's training and test accuracy
2. Enter customer details in the form provided
3. Click "Predict Churn" to see whether the customer is likely to churn

## Future Improvements
- Implement more advanced machine learning models (Random Forest, XGBoost, etc.)
- Add feature importance analysis to identify key factors affecting churn
- Incorporate cost-benefit analysis for retention strategies
- Enhance the web application with additional visualizations and insights
