from flask import Flask,render_template, request
import joblib
import numpy as  np
import pandas as pd
app = Flask(__name__)



@app.route("/",methods=["POST","GET"])
def prediction():
    Store = (request.form.get("Store"))
    Department= (request.form.get("Department"))
    Type = (request.form.get("Department_Type"))
    Size = (request.form.get("Department_Size"))
    Fuel_Price = (request.form.get("Fuel_Price"))
    Temprature = (request.form.get("Temprature"))
    CPI = (request.form.get("CPI"))
    Unemployment = (request.form.get("Unemployment"))
    Weekly_Sales_Lag = (request.form.get("Weekly_Sales_Lag"))

    XGBoost_model = joblib.load('test_walmart.pkl')
    result= XGBoost_model.predict(np.array([[Store, Department, Type, Size,Fuel_Price,Temprature,CPI,Unemployment,Weekly_Sales_Lag]],dtype=float))
    return render_template('index.html',
                                 prediction=result[0])

app.run(host="0.0.0.0",port=3000,debug=True)











