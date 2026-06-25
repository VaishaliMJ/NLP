"""--------------------------------------------------------------------------------------------------
                        Customer Prediction
                    Student Name:   Vaishali Jorwekar
Problem Statement:  Predict whether customer leave or continue with bank               
-------------------------------------------------------------------------------------------------"""
###########################################################################################
#   Imports
###########################################################################################
import os

import tensorflow as tf
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping,TensorBoard
import datetime
import pandas as pd

###########################################################################################
#   Constants
###########################################################################################
BORDER="-"*80
PROJECT_FOLDER = "ProjectData"
DATASET_FILENAME=f"{PROJECT_FOLDER}/Churn_Modelling.csv"
TEST_SIZE=0.2
RANDOM_STATE=42
LOG_DIR="logs/fit/"
MODEL_NAME='model.keras'
GENDER_LABEL_ENCODER="gender_label_encoder.pkl"
GEOGRAHY_ENCODER="geography_one_hot_encoder.pkl"
SCALAR_NAME="scalar.pkl"
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
    
###########################################################################################
#   Function        :   read_csv_file
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def read_csv_file(csvFileName=DATASET_FILENAME)->pd.DataFrame:
    dFrame=pd.read_csv(csvFileName)
    print(BORDER)
    print(f"Data loaded successfully from file '{csvFileName}'")
    print(BORDER)
    print(f"File Data:\n{BORDER}\n{dFrame.head}")
    print(f"Data Set Shape:{dFrame.shape}")
    print(f"Columns in data set:{dFrame.columns}")
    print(BORDER)
    return dFrame
###########################################################################################
#   Function        :   preprocess_data
#   Input Params    :   dataFrame(data frame)
#   Output Params   :   Updated Data Frame
#   Description     :   PreProcess Input data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def preprocess_data(dataFrame):
    #   Drop Columns
    dataFrame=dataFrame.drop(['RowNumber','CustomerId','Surname'],axis=1)
    
    #   Pre-Process Categorical feature
    gender_label_encoder=LabelEncoder()
    dataFrame['Gender']=gender_label_encoder.fit_transform(dataFrame['Gender'])
    
    #   One-Hot encoding for 'Geography' column
    geography_one_hot_encoder=OneHotEncoder()
    geo_encoding=geography_one_hot_encoder.fit_transform(dataFrame[['Geography']])
    geo_df=pd.DataFrame(geo_encoding.toarray(),
                        columns=geography_one_hot_encoder.get_feature_names_out(['Geography']))
    
    
    #   Drop 'Geography' column and add newly encoded columns
    dataFrame=pd.concat([dataFrame.drop('Geography',axis=1),geo_df],axis=1)
    print(dataFrame.head)
    
    #   Save Encoders
    save_encoders_scalar(gender_label_encoder,os.path.join(PROJECT_FOLDER,GENDER_LABEL_ENCODER))
    
    save_encoders_scalar(geography_one_hot_encoder,os.path.join(PROJECT_FOLDER,GEOGRAHY_ENCODER))

    return dataFrame
###########################################################################################
#   Function        :   save_encoders_scalar
#   Input Params    :   encoder
#   Output Params   :   None
#   Description     :   Save Encoders and scalars
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def save_encoders_scalar(encoder,encoderPath):
    with open(encoderPath,'wb') as file:
        pickle.dump(encoder,file)
###########################################################################################
#   Function        :   split_DataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def split_DataSet(dFrame): 
    X=dFrame.drop(['Exited'],axis=1)
    y=dFrame['Exited']  
    
    #   Split data into training and testing set
    xTrain,xTest,yTrain,yTest=train_test_split(X,y,
                                               test_size=TEST_SIZE,
                                               random_state=RANDOM_STATE)  
    #   Use Standard scalar
    scalar=StandardScaler()
    xTrain=scalar.fit_transform(xTrain)
    xTest=scalar.transform(xTest)
    
    #   Save scalar object
    save_encoders_scalar(scalar,os.path.join(PROJECT_FOLDER,SCALAR_NAME))

     
    return  xTrain,xTest,yTrain,yTest 
###########################################################################################
#   Function        :   build_Model
#   Input Params    :   xTrain
#   Output Params   :   Sequential Model
#   Description     :   Builded Model
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def build_Model(xTrain): 
    
    model=Sequential([
        #   First Hidden layer
        Dense(64,activation='relu',input_shape=(xTrain.shape[1],)),
        #   Second Hidden layer
        Dense(32,activation='relu'),
        #   Output Layer
        Dense(1,activation='sigmoid')
        
    ]   
    )
    model.summary()
    
    model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])
    
    return model
###########################################################################################
#   Function        :   setUp_Tensorboard
#   Input Params    :   None
#   Output Params   :   Log Directory
#   Description     :   Creates a log directory
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def setUp_Tensorboard(): 
    logDir=LOG_DIR+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorBoard=TensorBoard(log_dir=logDir,histogram_freq=1)
    
    return tensorBoard
###########################################################################################
#   Function        :   train_Model
#   Input Params    :   xTrain,xTest,yTrain,yTest,model,tensorBoard
#   Output Params   :   None
#   Description     :   Train built model
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def train_Model(xTrain,xTest,yTrain,yTest,model,tensorBoard): 
    #   Early stopping
    early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=4,
                    restore_best_weights=True
                )
    #   Train Model
    history=model.fit(
        xTrain,yTrain,
        validation_data=(xTest,yTest),
        epochs=100,
        callbacks=[early_stop,tensorBoard]
    )
    
    model.save(os.path.join(PROJECT_FOLDER,MODEL_NAME))

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
    #   Read CSV file
    df=read_csv_file()
    #   Pre-Process data
    df=preprocess_data(df)
    #   Split data set
    xTrain,xTest,yTrain,yTest=split_DataSet(df)
    #   Build model
    model=build_Model(xTrain)
    #   Set up tensorboard
    tensorBoard=setUp_Tensorboard()
    #   Train Model
    train_Model(xTrain,xTest,yTrain,yTest,model,tensorBoard)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()