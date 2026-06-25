"""--------------------------------------------------------------------------------------------------
                        Customer Prediction -Test Module
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

    # Test input data
    input_data = {
        'CreditScore': 600,
        'Geography': 'France',
        'Gender': 'Male',
        'Age': 40,
        'Tenure': 3,
        'Balance': 60000,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 50000
    }
    #   One hot encoding of "Geography"
    geo_encoded = geographyEncoder.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geographyEncoder.get_feature_names_out(['Geography']))
    
    input_df=pd.DataFrame([input_data])

    #   Gender Encoding
    input_df['Gender']=genderLabelEncoder.transform(input_df['Gender'])
    input_df=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)

    
    
    ## Scaling the input data
    input_scaled=scalar.transform(input_df)
    
    ## PRedict churn
    prediction=model.predict(input_scaled)
    
    prediction_proba = prediction[0][0]
    
    if prediction_proba > 0.5:
        print('The customer is likely to churn.')
    else:
        print('The customer is not likely to churn.')
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
