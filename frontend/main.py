# frontend/main.py
import requests
import streamlit as st
# defines an h1 header
st.title("is this a joke")

#link to about page
st.write("[about](https://credwood.substack.com/irony)")

# displays a text field
text = st.text_area('max. 500 words') 

# displays a button
if st.button('submit'):
    if text is not None:
        res = requests.post("http://backend:8080/test", json={"message":
                                                                    {"id": 0,
                                                                    "text": [text],
                                                                    "result":[],
                                                                    "softmax":[],
                                                                    "ground_truth":[5]
                                                                    }
                                                            }
                            )
        res = res.json()
        st.write(f"model response: {res.get("result")[0]}, probability: {res.get("softmax")[0]}")
        st.write(f"does the model's response of {res.get("result")[0]} seem correct?")
        yes = st.button("Yes")
        no = st.button("No")
        if yes:
            st.write("Thank you for your feedback!")
            res.ground_truth = [1]
            requests.post("http://backend:8080/test", json=res)
        elif no:
            st.write("Thank you for your feedback!")
            res.ground_truth = [0]
            requests.post("http://backend:8080/test", json=res)
            
    
