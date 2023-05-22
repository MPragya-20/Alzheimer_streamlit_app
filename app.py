import streamlit as st
from predict import predict
import numpy as np
 
st.title('Dementia prediction')
 
st.write('---')


 
st.markdown("**Please enter the correct details from the MRI report - **")

MR_Delay=st.number_input('Enter MR Delay')
gender = st.sidebar.selectbox('Gender (select 0 for male and 1 for female)',(0,1))
age = st.number_input('Age (in years)?')
EDUC=  st.number_input('Enter years of education')
SES= st.number_input('Enter Socio-economic Status (1-5)')
MMSE=  st.number_input('Enter Mini Mental State Examination (MMSE)')
CDR=  st.number_input('Enter Clinical Dementia Rating (CDR)')
eTIV=  st.number_input('Enter estimated total intracranial volume (eTIV)')
nWBV= st.number_input('Enter Normalize Whole Brain Volume (nWBV)')
ASF=st.number_input('Enter Atlas Scaling Factor (ASF)')
 
if st.button('Predict'):
    my_pred = predict(np.array([[(MR_Delay),	gender,     age,	EDUC,	SES,	MMSE,	CDR,	eTIV,	nWBV,	ASF]]))
    res = ""
    if my_pred[0]==0:
        res = "Non demented"
    else:
        res = "Demented"
    st.text(res)