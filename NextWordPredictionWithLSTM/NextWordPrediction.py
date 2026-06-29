"""--------------------------------------------------------------------------------------------------
                        Next Word Prediction
                    Student Name:   Vaishali Jorwekar
Problem Statement:  Next Word Prediction WIth LSTM            
-------------------------------------------------------------------------------------------------"""
###########################################################################################
#   Imports
###########################################################################################
import os


import tensorflow as tf
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import json


from keras.models import Sequential
from keras.layers import Embedding,Dense,LSTM,Dropout,Input
from keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences

import datetime
import pandas as pd
import numpy as np
###########################################################################################
#   Constants
###########################################################################################
BORDER="-"*80
PROJECT_FOLDER = "ProjectData"
DATASET_FILENAME=f"{PROJECT_FOLDER}/hamlet.txt"
TEST_SIZE=0.2
RANDOM_STATE=42
LOG_DIR="logs/fit/"
TOKENIZER_NAME="word_prediction_Tokenizer.json"
MODEL_NAME='nextWordPrediction.keras'
##########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Jun 2026
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   save_tokenizer_MaxSeqLen
#   Input Params    :   tokenizer,max_seq_len
#   Output Params   :   None
#   Description     :   Save and other data
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def save_tokenizer_MaxSeqLen(tokenizer,max_seq_len):
    tokenizer_data = {
    "tokenizer_config": tokenizer.to_json(),
    "max_seq_len": max_seq_len
    }

    with open(os.path.join(PROJECT_FOLDER,TOKENIZER_NAME),'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False)
###########################################################################################
#   Function        :   read_file
#   Input Params    :   dataSetFile
#   Output Params   :   Input file text
#   Description     :   Read input text file
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Jun 2026
############################################################################################
def read_file(fileName=DATASET_FILENAME):
    with open(fileName,"r") as file:
        text=file.read().lower()
    return text   
###########################################################################################
#   Function        :   tokenizeTextFile
#   Input Params    :   text
#   Output Params   :   Tokenized Text
#   Description     :   Tokenize Input text data
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Jun 2026
############################################################################################
def tokenizeTextFile(text): 
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts([text])
    totalWords=len(tokenizer.word_index)+1
    print(f"Total Words:{totalWords}")
    
    inputSequences=[]
    for line in text.split("\n"):
        token_list=tokenizer.texts_to_sequences([line])[0]
        for i in range(1,len(token_list)):
            nGramSeq=token_list[:i+1]
            inputSequences.append(nGramSeq)
    
    return inputSequences,totalWords,tokenizer
###########################################################################################
#   Function        :   padInputSequences
#   Input Params    :   inputSeq
#   Output Params   :   Padded Sequences
#   Description     :   Pad Input text Sequence
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Jun 2026
############################################################################################
def padInputSequences(inputSeq): 
    max_seq_len=max([len(x) for x in inputSeq])
    paddedInputSeq=np.array(pad_sequences(inputSeq,maxlen=max_seq_len,padding="pre"))
    return paddedInputSeq,max_seq_len 
###########################################################################################
#   Function        :   generateFeaturesAndLabels
#   Input Params    :   inputSeq,totalWords
#   Output Params   :   x,y
#   Description     :   Features and Labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Jun 2026
############################################################################################
def generateFeaturesAndLabels(inputSeq,totalWords):  
    x=inputSeq[:,:-1]
    y=inputSeq[:,-1]
    y=tf.keras.utils.to_categorical(y,num_classes=totalWords)
    
    return x,y
###########################################################################################
#   Function        :   split_DataSet
#   Input Params    :   x,y
#   Output Params   :   Training and testing dataset
#   Description     :   This method spilts data set into train and test dataset
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Jun 2026
############################################################################################
def split_DataSet(x,y): 
    
    
    #   Split data into training and testing set
    xTrain,xTest,yTrain,yTest=train_test_split(x,y,
                                               test_size=TEST_SIZE,
                                               random_state=RANDOM_STATE)  
    
    return  xTrain,xTest,yTrain,yTest 
###########################################################################################
#   Function        :   build_Model
#   Input Params    :   totalWords,max_seq_len
#   Output Params   :   Sequential Model
#   Description     :   Builded Model
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
############################################################################################
def build_Model(totalWords,max_seq_len): 
    model=Sequential()
    model.add(Input(shape=(max_seq_len-1,)))  

    model.add(Embedding(totalWords,100))  
    model.add(LSTM(150,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(totalWords,activation='softmax')) 
    
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model
###########################################################################################
#   Function        :   train_Model
#   Input Params    :   xTrain,xTest,yTrain,yTest,model
#   Output Params   :   None
#   Description     :   Train built model
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Jun 2026
############################################################################################
def train_Model(xTrain,xTest,yTrain,yTest,model): 
    #   Early stopping
    early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=15,
                    restore_best_weights=True
                )
    history=model.fit(
        xTrain,
        yTrain,
        epochs=100,
        validation_data=(xTest,yTest),
        verbose=1
         #callbacks=[early_stop]
         )
    model.save(os.path.join(PROJECT_FOLDER,MODEL_NAME))
    return history 
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Jun 2026
###############################################################################
def main():
    #   Project Folder
    ensure_dir(PROJECT_FOLDER)
    #   Read Input file
    text=read_file()
    #   Tokenize input file
    inputSequences,totalWords,tokenizer=tokenizeTextFile(text)
    
    #   Pad sequences
    paddedInputSeq,max_seq_len=padInputSequences(inputSequences)
    # Save Tokenizer and Max_Seq Len
    save_tokenizer_MaxSeqLen(tokenizer,max_seq_len)
    #   Genrate Features and lables
    x,y=generateFeaturesAndLabels(paddedInputSeq,totalWords)
    #   Split dataset
    xTrain,xTest,yTrain,yTest=split_DataSet(x,y)
    #   Build model
    model=build_Model(totalWords,max_seq_len)
    #   Train the model
    history=train_Model(xTrain,xTest,yTrain,yTest,model)
    
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
