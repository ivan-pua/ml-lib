import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='Telco Churn Prediction', version='1.0',
description='Adaboost model is used for prediction')

model = joblib.load("../model/Adaboost.pkl")


class Data(BaseModel):
    tenure: float
    Contract: int
    MonthlyCharges: float
    OnlineSecurity: int
    TechSupport: int
    PaperlessBilling: int

@app.get('/')

@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    """
    return {'message': 'System is healthy'}


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


@app.post("/predict")
def predict(data: Data):
    result = model.predict(np.array([data.tenure,
                            data.Contract,
                            data.MonthlyCharges,
                            data.OnlineSecurity,
                            data.TechSupport,
                            data.PaperlessBilling
                            ]).reshape(1,-1)
    )

    # fastapi does not recorgnise numpy, so have to covnert to list
    result = result.tolist()[0]

    if result == 1:
        return "Highly likely to churn"
    
    else:
        return "Not likely to churn"





