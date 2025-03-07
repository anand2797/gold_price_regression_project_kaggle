import os 
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model_trainer/model.pkl'
            preprocessor_path = 'artifacts/data_transformation/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            # transform the input data by using preprocessor
            transformed_data = preprocessor.transform(features)
            # predict the output by passing transformed data to model
            prediction = model.predict(transformed_data)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)
 
class CustomData:
    def __init__(self, **kwargs):
        """
        Initializes the CustomData object with input features dynamically.

        Args:
            **kwargs: Keyword arguments representing feature names and their values.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    

    def get_data_as_dataframe(self):
        """
        Converts the stored custom data attributes into a pandas DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the input features as columns
            with their corresponding values.
        Raises:
            CustomException: If an error occurs during DataFrame creation.
        """
        try:
            # Convert attributes to a dictionary
            custom_data_input = {key: [value] for key, value in self.__dict__.items()}

            # Create a DataFrame
            df = pd.DataFrame(custom_data_input)
            return df
        except Exception as e:
            raise CustomException(e, sys)

