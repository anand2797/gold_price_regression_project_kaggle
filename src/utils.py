import pandas as pd
import mysql.connector
# from pymongo import MongoClient
from src.logger import logging
from src.exception import CustomException
import sys
from dotenv import load_dotenv
import os

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
        query = os.getenv("MYSQL_QUERY")
        cursor = conn.cursor()

        cursor.execute(query)
        columns = [col[0] for col in cursor.description]  # Fetch column names
        data = cursor.fetchall()  # Fetch data

        df = pd.DataFrame(data, columns=columns)  # Create DataFrame

        cursor.close()
        conn.close()
        logging.info("Data successfully fetched from MySQL.")
        return df
    except Exception as e:
        raise CustomException(e, sys)
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