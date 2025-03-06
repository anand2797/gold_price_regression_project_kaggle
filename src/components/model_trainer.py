import os 
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import yaml

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@dataclass
class ModelTrainerConfig:
    model_filepath: str = os.path.join('artifacts', 'model_trainer', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def evaluate_model_performance(self, test_data, test_target, best_estimator):
        try:
            y_pred = best_estimator.predict(test_data)
            mse = mean_squared_error(y_true=test_target, y_pred=y_pred)
            mae = mean_absolute_error(y_true=test_target, y_pred=y_pred)
            r2 = r2_score(y_true=test_target, y_pred=y_pred)

            return mse, mae,r2
        except Exception as e:
            raise CustomException(e, sys)
        
    def perform_gridsearch(self, train_data, train_target, model, param_grid):
        try:
            
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                grid_search.fit(train_data, train_target)
                return  grid_search.best_estimator_, grid_search.best_params_
            return  model.fit(train_data, train_target), {}
                
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_transformed, y_train, test_transformed, y_test):
        """Runs the full model training process and saves the best model."""
        try:
            models = {
                "LinearRegression": (LinearRegression(), {}),
                "Ridge": (Ridge(), {"alpha": [0.1, 1, 10]}),
                "Lasso": (Lasso(), {"alpha": [0.001, 0.01, 0.1, 1]}),
                "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}),
                "SVR": (SVR(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
                "DecisionTree": (DecisionTreeRegressor(), {"max_depth": [3, 5, 10]}),
                #"RandomForest": (RandomForestRegressor(), {"n_estimators": [50, 100, 200]}),
                #"AdaBoost": (AdaBoostRegressor(), {"n_estimators": [50, 100, 200]})
            }

            best_model, best_score, best_name, best_params = None, float("-inf"), None, None

            
            # yaml file for put metics of all estimator
            yaml_filepath = "model_performance_results.yaml"
            # os.makedirs(os.path.dirname(yaml_filepath), exist_ok=True)

            # Clear YAML file before writing new results**
            with open(yaml_filepath, "w") as file:
                yaml.dump({}, file)

            results_dict = {"Train Performance":{}, "Test Performance":{}} # for each model metrics

            for name, (model, param_grid) in models.items():
                # Step 1: Perform Grid Search
                best_estimator, params = self.perform_gridsearch(train_transformed, y_train, model, param_grid)

                # Step :2.1 Evaluate Best Estimator for train data
                logging.info("..***...---------...***..")
                train_mse, train_mae, train_r2 = self.evaluate_model_performance(train_transformed, y_train, best_estimator)
                logging.info(f"Train Performance-->> {name}: MSE={train_mse}, MAE={train_mae}, R²={train_r2}")

                # Step 2.2: Evaluate Best Estimator for test data
                test_mse, test_mae, test_r2 = self.evaluate_model_performance(test_transformed, y_test, best_estimator)
                logging.info(f"Test Performance-->> {name}: MSE={test_mse}, MAE={test_mae}, R²={test_r2}")
                logging.info("..***...---------...***..")
                
                results_dict["Train Performance"].update({name: {"MSE": train_mse, "MAE": train_mae, "R²": train_r2}})
                results_dict["Test Performance"].update({name: {"MSE": test_mse, "MAE": test_mae, "R²": test_r2}})
                # **Write new results to YAML file**
                with open(yaml_filepath, "w") as file:
                    yaml.dump(results_dict, file, default_flow_style=False)

                # Step 3: Select Best Model
                if test_r2 > best_score:
                    best_model, best_score, best_name, best_params = best_estimator, test_r2, name, params
            logging.info("..***...---------...***..")
            logging.info(f"Best Model: {best_name} (R²={best_score:.4f}) | Params: {best_params}")
            logging.info("..***...---------...***..")
            
            # Step 4: Save Best Model
            save_object(self.model_trainer_config.model_filepath, best_model)

            #return {"best_model": best_name, "best_score": best_score, "best_params": best_params}
        except Exception as e:
            raise CustomException(e, sys)
