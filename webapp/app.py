import streamlit as st
st.title('Arabic Tweet Spam Classification')


with st.form("my_form"):
   st.text_input(label="Write a tweet and it will be classified!")
   
   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("submitted")
       
col1, col2, col3 = st.columns(3)

with col1:
   st.header("Naive Bayes")
   

with col2:
   st.header("SVM")
   

with col3:
   st.header("Bert")
   