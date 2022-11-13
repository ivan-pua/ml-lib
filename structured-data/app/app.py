import streamlit
import requests

def run():

    streamlit.title("Telco Churn Prediction")
    tenure = streamlit.number_input("Tenure")
    Contract = streamlit.number_input("Contract")
    MonthlyCharges = streamlit.number_input("Monthly Charges")
    OnlineSecurity = streamlit.number_input("Has Online Security?")
    TechSupport = streamlit.number_input("Tech Support")
    PaperlessBilling = streamlit.number_input("Has Paperless Billing?")

    data = {
        'tenure': tenure,
        'Contract': Contract,
        'MonthlyCharges': MonthlyCharges,
        'OnlineSecurity': OnlineSecurity,
        'TechSupport': TechSupport,
        'PaperlessBilling': PaperlessBilling,
        }

    if streamlit.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        prediction = response.text
        streamlit.success(f"The prediction from model: {prediction}")

if __name__ == '__main__':
    run()