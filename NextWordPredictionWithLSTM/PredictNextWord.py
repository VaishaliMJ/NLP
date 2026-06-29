"""--------------------------------------------------------------------------------------------------
                        Predict Next Word : Test Module
                    Student Name:   Vaishali Jorwekar
Problem Statement:  Next Word Prediction WIth LSTM  : Test module          
-------------------------------------------------------------------------------------------------"""
###########################################################################################
#   Imports
###########################################################################################
import os
import tensorflow as tf
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import json
import numpy as np
###########################################################################################
#   Constants
###########################################################################################
BORDER="-"*80
PROJECT_FOLDER = "ProjectData"
DATASET_FILENAME=f"{PROJECT_FOLDER}/hamlet.txt"
MODEL_NAME='nextWordPrediction.keras'
TOKENIZER_NAME="word_prediction_Tokenizer.json"
###########################################################################################
#   Function        :   load_tokenizer
#   Input Params    :   encoder
#   Output Params   :   tokenizer
#   Description     :   Load tokenizer
#   Author          :   Vaishali M Jorwekar
#   Date            :   29 Jun 2026
############################################################################################
def load_tokenizer(path):
   
    with open(path,'r') as file:
       return(json.load(file))
      
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
###############################################################################
#   Function        :   predict_next_word
#   Input Params    :   model,tokenizer,inputText,max_seq_len
#   Output Params   :   predicted word
#   Description     :   Prediction Word using Pre-Trained Model
#   Author          :   Vaishali M Jorwekar
#   Date            :   29 Jun 2026
###############################################################################
def predict_next_word(model,tokenizer,inputText,max_seq_len):
    tokenList=tokenizer.texts_to_sequences([inputText])[0]
    if len(tokenList) >= max_seq_len:
        tokenList=tokenList[-(max_seq_len-1):]
    tokenList=pad_sequences([tokenList],maxlen=max_seq_len-1,padding='pre')
    predicted=model.predict(tokenList,verbose=0)
    predictedWordIndex=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predictedWordIndex:
            return word
          
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   29 Jun 2026
###############################################################################
def main():
    #   Project Folder
    ensure_dir(PROJECT_FOLDER)
    model=load_model(os.path.join(PROJECT_FOLDER,MODEL_NAME))
    tokenizerData=load_tokenizer(os.path.join(PROJECT_FOLDER,TOKENIZER_NAME))
    tokenizer=tokenizer_from_json(tokenizerData["tokenizer_config"])
    max_seq_len=tokenizerData["max_seq_len"]
    
    inputText="Barn. Well, goodnight. If you do "
    
    wordPredicted=predict_next_word(model,tokenizer,inputText,max_seq_len)
    print(f"Input sentence : {inputText}")
    print(f"Predicted Word  : {wordPredicted}")
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
