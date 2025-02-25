import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str  = os.path.join('artifacts','data_ingestion', 'train.csv')
    test_data_path: str = os.path.join('artifacts','data_ingestion', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data_ingestion', 'raw.csv')

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self):
        logging.info('>>> Entered the initiate data ingestion method... <<<')
        try:
            df = pd.read_csv('notebooks/data/cleaned_data.csv')
            logging.info('REad the dataset as dataframe...')
            
            data_ingestion_dir_name = os.path.dirname(self.data_ingestion_config.train_data_path) # dir name
            os.makedirs(data_ingestion_dir_name, exist_ok=True) # make dir as 'artifacts/data_ingestion'

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=None) # save raw data

            logging.info("Train test split initiated: ")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save train data in csv
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header = None)
            # save test data in csv
            test_set.to_csv(self.data_ingestion_config.test_data_path, index = False, header = None)

            logging.info(">>> Data Ingestion is Completed... <<<")

            return self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
        

"""
if __name__ == "__main__":
    obj = DataIngestion(data_ingestion_config=DataIngestionConfig())
    obj.initiate_data_ingestion()
"""
