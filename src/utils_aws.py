import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from dotenv import load_dotenv
load_dotenv()

import boto3
import pickle
import pickle
import pickle

# Get AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# Create a single S3 client
s3_client = boto3.client("s3", 
                         aws_access_key_id=AWS_ACCESS_KEY, 
                         aws_secret_access_key=AWS_SECRET_KEY, 
                         region_name=AWS_REGION)

# create a fuction to fetch model from s3 bucket and load it by pickle 
def fetch_model_from_s3(model_name, local_path, bucket_name=BUCKET_NAME):
    """
    Fetches a model from an S3 bucket and loads it using pickle.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        model_name (str): The name of the model file in the bucket.
        local_path (str): The local path where the model will be temporarily saved.
    Returns:
        any: The loaded model object.
    Raises:
        CustomException: If an error occurs while fetching the model.
    """
    try:
        # Download the model from S3 and save locally
        s3_client.download_file(bucket_name, model_name, local_path)

        # Load the model using pickle
        with open(local_path, "rb") as f:
            model = pickle.load(f)
        #logging.info(f"Model '{model_name}' successfully fetched from S3 bucket '{bucket_name}'.")
        return model
    except Exception as e:
        raise CustomException(e, sys)
    
# create a fuction to push model to s3 bucket 
def push_model_to_s3(model_name, local_path, bucket_name=BUCKET_NAME):
    """
    Pushes a model to an S3 bucket.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        model_name (str): The name of the model file in the bucket.
        local_path (str): The local path where the model will be temporarily saved.
    Returns:
        None
    Raises:
        CustomException: If an error occurs while pushing the model.
    """
    try:
        # Upload the model to S3
        s3_client.upload_file(local_path, model_name, bucket_name=BUCKET_NAME)
        logging.info(f"Model '{model_name}' successfully pushed to S3 bucket '{bucket_name}'.")
    except Exception as e:
        raise CustomException(e, sys)



