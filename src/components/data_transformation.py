import os 
import sys 
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,  FunctionTransformer, PowerTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass



@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath: str = os.path.join('artifacts', 'data_transformation', 'preprocessor.pkl')
    # "artifacts/data_transforamtion/preprocessor.pkl"
    train_datapath: str = os.path.join("artifacts", 'data_ingestion', 'train.csv')
   # test_scaled_datapath: str = os.path.join("artifacts", 'data_transformation','test_scaled.npy')

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig ):
        self.data_transformation_config = DataTransformationConfig()

    def get_features_by_transformation_methods(self, df: pd.DataFrame):
        
        # apply skew on each numerical col of df, then give transforamtion method based on it, 
        # then create list as transformation method feature and append column name in it , 
        # for ex. boxcox_features=[], yen_johnson_features=[], log_features = []
        # returns these features in a dict
        # use this fuction in following fuction to get features list, then perform 

        """
        Categorizes numerical columns into different transformation types based on skewness.

        Parameters:
        df (DataFrame): The input dataframe containing numerical columns.

        Returns:
        dict: A dictionary containing lists of column names categorized by transformation type.
        """
        # Select only numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Initialize empty lists
        boxcox_features = []
        log_features = []
        yeo_johnson_features = []

        for col in num_cols:
            skewness = df[col].skew()
            min_value = df[col].min()

            if -0.5 <= skewness <= 0.5 and min_value > 0:
                boxcox_features.append(col)  # Box-Cox for low-skew, all positive
            elif abs(skewness) > 0.5:
                if skewness > 1 and min_value > 0:
                    log_features.append(col)  # Log1p for right-skewed, all positive
                elif skewness < -1:
                    if min_value > 0:
                        boxcox_features.append(col)  # Box-Cox for left-skewed, all positive
                    else:
                        yeo_johnson_features.append(col)  # Yeo-Johnson for left-skewed, mixed
                else:
                    yeo_johnson_features.append(col)  # Yeo-Johnson for general high-skew mixed

        return {
            "boxcox_features": boxcox_features,
            "log_features": log_features,
            "yeo_johnson_features": yeo_johnson_features,
            "all_num_features": num_cols
             }



    def get_data_transformer_object(self, train_data):
        try:
            
            # ColumnTransformer to apply transformations
            features = self.get_features_by_transformation_methods(df= train_data)

            # imputer
            imputer = SimpleImputer(strategy='median')
            transformer = ColumnTransformer([
                ('boxcox', PowerTransformer(method='box-cox'), features["boxcox_features"]),
                ('log', FunctionTransformer(np.log1p), features["log_features"]),
                ('yeo_johnson', PowerTransformer(method='yeo-johnson'), features["yeo_johnson_features"])
            ], remainder='passthrough')

            # Ensure transformer outputs DataFrame
            transformer.set_output(transform="pandas")  # <-- Add this

            # Final pipeline: Imputation → Transformation → Scaling
            preprocessor = Pipeline([
                                     #('imputer', imputer),     # Step 1: Handle missing values
                                     ('transformer', transformer),  # Step 2: Apply transformations
                                     ('scaler', StandardScaler())  # Step 3: Standard Scaling
                                    ])
            
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('>>> Read train and test data competed: <<<')

            logging.info(">>> Obtaining preprcesor object")
            target_column = 'gold close'
            input_train_data = train_df.drop(target_column, axis=1)
            target_train = train_df[target_column]

            input_test_data = test_df.drop(target_column, axis=1)
            target_test = test_df[target_column]

            print(type(input_train_data))
            preprocessor_obj = self.get_data_transformer_object(input_train_data)
            logging.info(">> Apply preprocessor object on training and test data..<<")
            # print(input_train_data.columns)
            # **Fit the pipeline on train data**
            preprocessor_obj.fit(input_train_data)

            # **Transform both train and test data**
            train_transformed = preprocessor_obj.transform(input_train_data)
            test_transformed = preprocessor_obj.transform(input_test_data)

            # **Save preprocessor as .pkl**
            save_object(self.data_transformation_config.preprocessor_obj_filepath, preprocessor_obj)

            logging.info(">>> Preprocessor fitted on train data and saved successfully. <<<")
            
            return {
                    "X_train": train_transformed,
                    "y_train": target_train,
                    "X_test": test_transformed,
                    "y_test": target_test,
                    "preprocessor": preprocessor_obj
                    }

        except Exception as e:
            raise CustomException(e, sys)


        


