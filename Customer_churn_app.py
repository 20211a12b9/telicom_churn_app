# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:12:09 2024

@author: chowd
"""
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the DecisionTreeClassifier model
load_data = pickle.load(open('C:/Users/chowd/Downloads/Customer_churn.sav', 'rb'))

# Initialize LabelEncoder
label_encoder = LabelEncoder()

def clean_input(input_data):
    # Convert input data to appropriate types and handle missing values
    cleaned_data = []
    for col, value in input_data.items():
        if col in ['SeniorCitizen', 'Tenure', 'MonthlyCharges', 'TotalCharges']:
            # Convert numerical values to float
            if value == '':
                cleaned_data.append(np.nan)  # Replace empty string with NaN
            else:
                cleaned_data.append(float(value))
        else:
            # Encode categorical values using LabelEncoder
            cleaned_data.append(label_encoder.fit_transform([value])[0])
    return cleaned_data

def Customer_churn_analysis(input_data):
    input_data_np = np.asarray(input_data, dtype=np.int32)
    input_data_reshaped = input_data_np.reshape(1, -1)
    prediction = load_data.predict(input_data_reshaped)
    return prediction[0]

def main():
    st.title('Customer Churn Analysis')
    
    # Input fields
    input_data = {
        'Gender': st.text_input('Gender'),
        'SeniorCitizen': st.text_input('Senior Citizen'),
        'Partner': st.text_input('Partner'),
        'Dependents': st.text_input('Dependents'),
        'Tenure': st.text_input('Tenure'),
        'PhoneService': st.text_input('Phone Service'),
        'MultipleLines': st.text_input('Multiple Lines'),
        'InternetService': st.text_input('Internet Service'),
        'OnlineSecurity': st.text_input('Online Security'),
        'OnlineBackup': st.text_input('Online Backup'),
        'DeviceProtection': st.text_input('Device Protection'),
        'TechSupport': st.text_input('Tech Support'),
        'StreamingTV': st.text_input('Streaming TV'),
        'StreamingMovies': st.text_input('Streaming Movies'),
        'Contract': st.text_input('Contract'),
        'PaperlessBilling': st.text_input('Paperless Billing'),
        'PaymentMethod': st.text_input('Payment Method'),
        'MonthlyCharges': st.text_input('Monthly Charges'),
        'TotalCharges': st.text_input('Total Charges')
    }
   
    # Predict churn button
    if st.button('Predict Churn'):
        cleaned_input = clean_input(input_data)
        churn_prediction = Customer_churn_analysis(cleaned_input)
        if churn_prediction == 0:
            st.success('No Churn')
        else:
            st.error('Churn Detected')

if __name__ == '__main__':
    main()

    main()
