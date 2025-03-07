from flask import Flask, request, render_template, jsonify
import pandas as pd 
import numpy as np
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import sys

import pickle

app = Flask(__name__)

# route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict():
    try:
        if request.method=='GET':
            return render_template('home.html')
        else:
            # Get JSON data from request body
            data= request.json 
            # Unpack dictionary into CustomData class
            custom_data = CustomData(**data)  

            input_df =  custom_data.get_data_as_dataframe()
            print(input_df)

            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_df)
            output = prediction[0]

            # Return JSON response for Postman
            return jsonify({"prediction": output})

             # If rendering on HTML page (Uncomment below code)
             # return render_template("result.html", prediction_text=f"Predicted Value: {output}")


    except Exception as e:
        raise CustomException(e, sys)
    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", use_reloader=False)

