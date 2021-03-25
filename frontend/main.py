# frontend/main.py
import requests
import streamlit as st

# defines an h1 header
st.title("is this a joke")

# displays a file uploader widget
text = st.text_input('Input your sentence here:') 

# displays a button
if text:
    res = requests.post(f"http://backend:8080/test", test.text=[text])
    st.write(f"model response: {res.result[0]}, probability: {res.softmax[0]}")
    st.write(f"does the model's response of {res.result[0]} seem correct?")
    yes = st.button("Yes")
    no = st.button("No")
    if yes:
        st.write(f"Thank you for your feedback!")
        res.ground_truth = [1]
        requests.post(f"http://backend:8080/test", test=res)
    elif no:
        res.ground_truth = [0]
        requests.post(f"http://backend:8080/test", test=res)
        st.write(f"Thank you for your feedback!")
    
