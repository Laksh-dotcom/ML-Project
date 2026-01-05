import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from src.logger import logging

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifiacts", "model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Entered Model Training.")
            X_train, Y_train, X_test, Y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "linear": LinearRegression(),
                "k-neighbors Regressor": KNeighborsRegressor(),
                "decision tree": DecisionTreeRegressor(),
                "random forest": RandomForestRegressor(),
                "adaboost": AdaBoostRegressor(),    
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "gradientBoosting": GradientBoostingRegressor()
            }
            model_report:dict = evaluate_model(x_train = X_train, y_train =  Y_train, x_test = X_test, y_test = Y_test, models = models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info("Model Training Done.")
            save_object(
                file_path = self.model_training_config.trained_model_file_path,
                obj = best_model
            )

            y_predicted = best_model.predict(X_test)
            r2_value = r2_score(Y_test, y_predicted)
            return r2_value
        
        except Exception as e:
            raise CustomException(e, sys)