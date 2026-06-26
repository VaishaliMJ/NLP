"""--------------------------------------------------------------------------------------------------
                        Customer Prediction -Streamlit Test Module
                    Student Name:   Vaishali Jorwekar
Problem Statement:  Predict whether customer leave or continue with bank               
-------------------------------------------------------------------------------------------------"""
###########################################################################################
#   Imports
###########################################################################################
import os
import tensorflow as tf
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from keras.models import load_model
import pickle
import numpy as np
import pandas as pd
import streamlit as st

###########################################################################################
#   Constants
###########################################################################################
BORDER="-"*80
PROJECT_FOLDER = "ProjectData"
MODEL_NAME='model.keras'
GENDER_LABEL_ENCODER="gender_label_encoder.pkl"
GEOGRAHY_ENCODER="geography_one_hot_encoder.pkl"
SCALAR_NAME="scalar.pkl"

###########################################################################################
#   Function        :   load_encoders_scalar
#   Input Params    :   encoder
#   Output Params   :   None
#   Description     :   Load Encoders and scalars
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def load_encoders_scalar(encoderPath):
   
    with open(encoderPath,'rb') as file:
       return(pickle.load(file))
      
##########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True) 
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
###############################################################################
def main():
    #   Project Folder
    ensure_dir(PROJECT_FOLDER)
    model=load_model(os.path.join(PROJECT_FOLDER,MODEL_NAME))
    scalar=load_encoders_scalar(os.path.join(PROJECT_FOLDER,SCALAR_NAME))
    genderLabelEncoder=load_encoders_scalar(os.path.join(PROJECT_FOLDER,GENDER_LABEL_ENCODER))
    geographyEncoder=load_encoders_scalar(os.path.join(PROJECT_FOLDER,GEOGRAHY_ENCODER))
    #CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited

    st.title('Customer Churn Prediction')

    # User input
    geography = st.selectbox('Geography', geographyEncoder.categories_[0])
    gender = st.selectbox('Gender', genderLabelEncoder.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])
    # Test input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [genderLabelEncoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    #   One hot encoding of "Geography"
    geo_encoded = geographyEncoder.transform([[geography]]).toarray()

    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geographyEncoder.get_feature_names_out(['Geography']))
    
    
    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scalar.transform(input_data)

    

    
    
    ## Scaling the input data
    input_scaled=scalar.transform(input_data_scaled)
    
    ## PRedict churn
    prediction=model.predict(input_scaled)
    
    prediction_proba = prediction[0][0]
    
    st.write(f'Churn Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
