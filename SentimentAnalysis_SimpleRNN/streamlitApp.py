import numpy as np
import os
import tensorflow as tf
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import streamlit as st


word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
###############################################################################
def main():
   
    model=load_model('simple_rnn_model.h5')
   
    
    st.title("IMDB Moview : Sentiment Analysis")
    user_input=st.text_area('Movie Review')
    
    if st.button("Predict Sentiment"):
        preprocess_input=preprocessInputText(user_input)
        
        
        prediction=model.predict(preprocess_input)
        
        sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
        st.write(f"Sentiment:{sentiment}")
        st.write(f"Prediction Score:{prediction[0][0]}")
    else:
        st.write("Enter a movie review")    
###############################################################################
#   Function        :   decode_review
#   Input Params    :   encoded_review
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
###############################################################################
def decode_review(encoded_review):
  
  return ' '.join([reverse_word_index.get(word-3,'?') for word in encoded_review])

###############################################################################
#   Function        :   preprocessInputText
#   Input Params    :   text
#   Output Params   :   padded review
#   Description     :   Input text with padded sequences
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Jun 2026
###############################################################################
def preprocessInputText(text):
  words=text.lower().split()
  
  encoded_review=[word_index.get(word,2)+3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()