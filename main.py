import os 
import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    # data ingestion step
    data_ingestion_config = DataIngestionConfig()
    data_ingestion_obj = DataIngestion(data_ingestion_config=data_ingestion_config)
    logging.info(">>>> Data ingestion Started... <<<<")
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()
    logging.info(">>>> Data Ingestion Completed....<<<<")

    # data Transformation step
    data_transformation_config = DataTransformationConfig()
    data_transformation_obj = DataTransformation(data_transformation_config=data_transformation_config)
    logging.info(">>>> Data Transformation Started.....<<<<")
    transformed_data=data_transformation_obj.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
    X_train = transformed_data["X_train"]
    y_train = transformed_data["y_train"]
    X_test = transformed_data["X_test"]
    y_test = transformed_data["y_test"]
    preprocessor = transformed_data["preprocessor"]
    logging.info(">>>> Data Transformation Completed...<<<<")

    model_trainer = ModelTrainer()
    logging.info(">>>> Model Training is started....<<<<")
    model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
    logging.info(">>>> Model Training is completed....<<<<")
