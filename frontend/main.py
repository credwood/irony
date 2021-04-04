# frontend/main.py
import requests
import streamlit as st
import SessionState
import json

# keeping track of res for button push state changes
session_state = SessionState.get(res=None)

# defines an h1 header
st.title("is this a joke")

#link to about page
st.write("[about](https://credwood.substack.com/p/irony)")

# displays a text field
text = st.text_area('max. 500 words') 

def convert_json(res):
    result = json.dumps(res)
    return result

def load_data(res):
    data = requests.post("http://backend:8084/test", json=res)
    return data.json()

# displays a button
if st.button('submit'):
    if text is not None:
        res = load_data({
                            "id": 0,
                            "text": [text],
                            "result":[],
                            "softmax":[],
                            "ground_truth":[5]
                        }
            )
        session_state.res = res
        st.write(f"model response: {res.get('result')[0]}, probability: {res.get('softmax')[0]}")
        st.write(f"does the model's response of {res.get('result')[0]} seem correct?")

if session_state.res is not None:
    yes = st.button("Yes")
    no = st.button("No")
    if yes:
        res = session_state.res
        res["ground_truth"]= [1]
        rsp=requests.post("http://backend:8084/test", json=res)
        st.write(f"Thank you for your feedback! Your response will be used to improve the model.")
    elif no:
        res = session_state.res
        res["ground_truth"]=[0]
        rsp=requests.post("http://backend:8084/test", json=res)
        st.write(f"Thank you for your feedback! Your response will be used to improve the model.")


