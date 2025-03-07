import pandas as pd
import mysql.connector
# from pymongo import MongoClient
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import sys
from dotenv import load_dotenv
import os

import pickle
import joblib

load_dotenv()

def read_from_mysql():
    """Reads data from MySQL and returns a DataFrame."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()
        query = os.getenv("MYSQL_QUERY") # select * from table_name
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]  # Fetch column names
        data = cursor.fetchall()  # Fetch data
        df = pd.DataFrame(data, columns=columns)  # Create DataFrame
        logging.info("Data successfully fetched from MySQL.")
        return df
    except Exception as e:
        raise CustomException(e, sys)
    finally:
        cursor.close()
        conn.close()
"""
def read_from_mongodb():
    # Reads data from MongoDB Atlas and returns a DataFrame.
    try:
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client[os.getenv("MONGO_DATABASE")]
        collection = db[os.getenv("MONGO_COLLECTION")]
        data = list(collection.find())
        df = pd.DataFrame(data)
        client.close()
        if '_id' in df.columns:
            df.drop(columns=['_id'], inplace=True)
        logging.info("Data successfully fetched from MongoDB Atlas.")
        return df
    except Exception as e:
        raise CustomException(e, sys)
"""

def save_object(file_path, obj, use_joblib=False):
    """
    Saves an object to a file using pickle or joblib.

    Parameters:
        file_path (str): 
            The path where the object should be saved.
        obj (any): 
            The object to save.
        use_joblib (bool): 
            If True, saves using joblib. Otherwise, uses pickle.
    Returns:
        None
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if use_joblib:
            joblib.dump(obj, file_path)  # Save using joblib
            logging.info(f"Object successfully saved using joblib at: {file_path}")
        else:
            with open(file_path, 'wb') as file:
                pickle.dump(obj, file)  # Save using pickle
            logging.info(f"Object successfully saved using pickle at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

# create a fuction to load a model or object which is stored iside pkl file or joblib   
def load_object(file_path, use_joblib=False):
    """
    Loads an object from a file using pickle or joblib.
    Parameters:
        file_path (str): The path from which the object should be loaded.
        use_joblib (bool): If True, loads using joblib. Otherwise, uses pickle.
    Returns:
        any: The loaded object.
    Raises:
        CustomException: If an error occurs while loading the object.
    """
    try:
        if use_joblib:
            return joblib.load(file_path)  # Load using joblib
        else:
            with open(file_path, 'rb') as file:
                return pickle.load(file)  # Load using pickle
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model_performance(test_data, test_target, best_estimator):
    """
    Evaluates the performance of a trained model using Mean Squared Error (MSE), 
    Mean Absolute Error (MAE), and R-squared (R²) score.
    Args:
        test_data (array-like or DataFrame): The feature data for testing.
        test_target (array-like or Series): The actual target values for testing.
        best_estimator (object): The trained model used for making predictions.
    Returns:
        tuple: A tuple containing three values -
            - mse (float): Mean Squared Error.
            - mae (float): Mean Absolute Error.
            - r2 (float): R² score.
    Raises:
        CustomException: If an error occurs during model evaluation.
    """
    try:
        y_pred = best_estimator.predict(test_data)
        mse = mean_squared_error(y_true=test_target, y_pred=y_pred)
        mae = mean_absolute_error(y_true=test_target, y_pred=y_pred)
        r2 = r2_score(y_true=test_target, y_pred=y_pred)

        return mse, mae,r2
    except Exception as e:
        raise CustomException(e, sys)

