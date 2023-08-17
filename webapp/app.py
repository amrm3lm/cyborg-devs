from models import SVMModel, LRModel
from bert_model import BERTModel
import os

import streamlit as st
st.title('Arabic Tweet Spam Classification')


@st.cache_data
def svm_model():
   return SVMModel()


@st.cache_data
def lr_model():
   return LRModel()


@st.cache_resource
def bert_model():
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
   return BERTModel()



# @st.cache_data
# def dt_model():
#    return DTModel()

bert_model()

with st.form("my_form"):
   inp = st.text_area(label="Write a tweet and it will be classified!")
   
   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("submitted")
       svm_res = svm_model().predict(inp)
       lr_res = lr_model().predict(inp)
       bert_res = bert_model().predict(inp)
      #  dt_res = dt_model().predict(inp)

       
       
col1, col2, col3= st.columns(3)  

with col1:
   st.header("Linear Regression")
   try:
      st.write(lr_res)
   except NameError:
      pass


with col2:
   st.header("SVM")
   
   try:
      st.write(svm_res)
   except NameError:
      pass

with col3:
   st.header("Bert")
   try:
      st.write(bert_res)
   except NameError:
      pass
