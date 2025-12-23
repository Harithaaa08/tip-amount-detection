import streamlit as st
import requests

st.title("Tip Amount Prediction App")
st.write("Enter the details below to predict the tip amount:")

# Input fields
total_bill = st.number_input("Total Bill Amount", min_value=0.0)
sex = st.selectbox("Sex", options=['Male','Female'])
smoker = st.selectbox("Smoker", options=['Yes','No'])
day = st.selectbox("Day of the Week", options=['Thur','Fri','Sat','Sun'])
time = st.selectbox("Time of Day", options=['Lunch','Dinner'])
size = st.number_input("Size of the Party", min_value=1, step=1)

if st.button("Predict Tip Amount"):
    # Prepare data
    input_data = {
        'total_bill': total_bill,
        'sex': sex,
        'smoker': smoker,
        'day': day,
        'time': time,
        'size': size
    }

    try:
        # Call Flask API (note correct port)
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)

        if response.status_code == 200:
            prediction = response.json().get('predicted_tip')
            st.success(f"The predicted tip is: {prediction:.2f}")
        else:
            st.error(f"Error in prediction: {response.json().get('error')}")

    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
