import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set up Streamlit page
st.set_page_config(page_title="Telecom Customer Churn Predictor", layout="centered")
st.title("📞 Telecom Customer Churn Predictor")


# Load and preprocess data
df = pd.read_csv('Telco-Customer-Churn.csv')


# Drop non-useful column
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric and drop missing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

data = df.copy()

# Encode categorical columns
cols_to_encode = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'Churn'
]
label_encoders = {}
for col in cols_to_encode:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target
y = data['Churn']
X = data.drop('Churn', axis=1)

# Scale numerical features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Model training
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

st.subheader("Model Performance")
st.write(f"- Training Accuracy: **{train_acc:.2f}**")
st.write(f"- Test Accuracy: **{test_acc:.2f}**")

# User Input Section
st.subheader("Enter Customer Details to Predict Churn:")

def user_input_features():
    gender = st.selectbox('Gender', options=label_encoders['gender'].classes_)
    SeniorCitizen = st.selectbox('Senior Citizen', options=[0, 1])
    Partner = st.selectbox('Partner', options=label_encoders['Partner'].classes_)
    Dependents = st.selectbox('Dependents', options=label_encoders['Dependents'].classes_)
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    PhoneService = st.selectbox('Phone Service', options=label_encoders['PhoneService'].classes_)
    MultipleLines = st.selectbox('Multiple Lines', options=label_encoders['MultipleLines'].classes_)
    InternetService = st.selectbox('Internet Service', options=label_encoders['InternetService'].classes_)
    OnlineSecurity = st.selectbox('Online Security', options=label_encoders['OnlineSecurity'].classes_)
    OnlineBackup = st.selectbox('Online Backup', options=label_encoders['OnlineBackup'].classes_)
    DeviceProtection = st.selectbox('Device Protection', options=label_encoders['DeviceProtection'].classes_)
    TechSupport = st.selectbox('Tech Support', options=label_encoders['TechSupport'].classes_)
    StreamingTV = st.selectbox('Streaming TV', options=label_encoders['StreamingTV'].classes_)
    StreamingMovies = st.selectbox('Streaming Movies', options=label_encoders['StreamingMovies'].classes_)
    Contract = st.selectbox('Contract', options=label_encoders['Contract'].classes_)
    PaperlessBilling = st.selectbox('Paperless Billing', options=label_encoders['PaperlessBilling'].classes_)
    PaymentMethod = st.selectbox('Payment Method', options=label_encoders['PaymentMethod'].classes_)
    MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    TotalCharges = st.number_input('Total Charges', min_value=0.0, max_value=15000.0, value=1000.0, step=1.0)

    # Encode inputs
    features = {
        'gender': label_encoders['gender'].transform([gender])[0],
        'SeniorCitizen': SeniorCitizen,
        'Partner': label_encoders['Partner'].transform([Partner])[0],
        'Dependents': label_encoders['Dependents'].transform([Dependents])[0],
        'tenure': tenure,
        'PhoneService': label_encoders['PhoneService'].transform([PhoneService])[0],
        'MultipleLines': label_encoders['MultipleLines'].transform([MultipleLines])[0],
        'InternetService': label_encoders['InternetService'].transform([InternetService])[0],
        'OnlineSecurity': label_encoders['OnlineSecurity'].transform([OnlineSecurity])[0],
        'OnlineBackup': label_encoders['OnlineBackup'].transform([OnlineBackup])[0],
        'DeviceProtection': label_encoders['DeviceProtection'].transform([DeviceProtection])[0],
        'TechSupport': label_encoders['TechSupport'].transform([TechSupport])[0],
        'StreamingTV': label_encoders['StreamingTV'].transform([StreamingTV])[0],
        'StreamingMovies': label_encoders['StreamingMovies'].transform([StreamingMovies])[0],
        'Contract': label_encoders['Contract'].transform([Contract])[0],
        'PaperlessBilling': label_encoders['PaperlessBilling'].transform([PaperlessBilling])[0],
        'PaymentMethod': label_encoders['PaymentMethod'].transform([PaymentMethod])[0],
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    return pd.DataFrame([features])

input_df = user_input_features()

# Scale numerical features in input
def scale_input(df_input):
    df_scaled = df_input.copy()
    df_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']])
    return df_scaled

input_scaled = scale_input(input_df)

if st.button('Predict Churn'):
    pred = model.predict(input_scaled)
    result = label_encoders['Churn'].inverse_transform(pred)[0]
    st.success(f"The customer is likely to: **{result}**")
