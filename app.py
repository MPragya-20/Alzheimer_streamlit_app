import streamlit as st
from predict import predict
import numpy as np
 
st.title('Dementia prediction')
 
st.write('---')
 

 
st.markdown("**Please enter the details in the form**")

MR_Delay=st.number_input('Enter MR_Delay')
gender = st.sidebar.selectbox('gender',(0,1))
#area = st.slider('Area of the house', 1000, 5000, 1500)
age = st.number_input('Age (in years)?')
EDUC=  st.number_input('Enter EDUC')
SES= st.number_input('Enter SES')
MMSE=  st.number_input('Enter MMS')
CDR=  st.number_input('Enter CDR')
eTIV=  st.number_input('Enter eTIV')
nWBV= st.number_input('Enter nWBV')
ASF=st.number_input('Enter ASF')
 
if st.button('Predict'):
    my_pred = predict(np.array([[(MR_Delay),	gender,     age,	EDUC,	SES,	MMSE,	CDR,	eTIV,	nWBV,	ASF]]))
    res = ""
    if my_pred[0]==0:
        res = "Non demented"
    else:
        res = "Demented"
    st.text(res)