import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# بارگذاری مدل و ستون‌های مدل
model = joblib.load('best_model.pkl')
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.title("💡 Monthly Charges Prediction App")

st.markdown("""
وارد کردن اطلاعات مشتری برای پیش‌بینی مبلغ شارژ ماهانه.
""")

# فرم ورودی
gender = st.selectbox('Gender', ['Female', 'Male'])
SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
Partner = st.selectbox('Partner', ['No', 'Yes'])
Dependents = st.selectbox('Dependents', ['No', 'Yes'])
tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12)

PhoneService = st.selectbox('Phone Service', ['No', 'Yes'])
MultipleLines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
OnlineBackup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
DeviceProtection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
TechSupport = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
StreamingMovies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox('Paperless Billing', ['No', 'Yes'])
PaymentMethod = st.selectbox('Payment Method', [
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)'
])

# آماده‌سازی دیتافریم
input_dict = {
    'SeniorCitizen': SeniorCitizen,
    'tenure': tenure,
    'gender': gender,
    'Partner': Partner,
    'Dependents': Dependents,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod
}

input_df = pd.DataFrame([input_dict])

# تابع پردازش
def preprocess(df):
    df_processed = df.copy()
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)
    df_processed['PaperlessBilling_Yes'] = (df_processed['PaperlessBilling'] == 'Yes').astype(int)
    df_processed.drop(columns=['PaperlessBilling'], inplace=True)

    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    df_processed = pd.get_dummies(df_processed, columns=cat_cols)

    for col in model_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    df_processed = df_processed[model_columns]

    return df_processed

# دکمه پیش‌بینی
if st.button('🔮 Predict Monthly Charges'):
    X = preprocess(input_df)
    prediction = model.predict(X)[0]
    st.success(f"💰 Predicted Monthly Charges: ${prediction:.2f}")

    # مقایسه با دیتای واقعی
    try:
        df_real = pd.read_csv('data_modified.csv')
        mean_price = df_real['MonthlyCharges'].mean()
        std_price = df_real['MonthlyCharges'].std()

        delta = prediction - mean_price
        st.write(f"📊 **Average MonthlyCharges in dataset:** ${mean_price:.2f}")
        st.write(f"📉 Your prediction is {'above' if delta > 0 else 'below'} average by ${abs(delta):.2f}")
        st.info(f"ℹ️ Typical range: ${mean_price - std_price:.2f} - ${mean_price + std_price:.2f}")

    except Exception as e:
        st.warning("⚠️ Could not load dataset for comparison.")
        st.error(str(e))
