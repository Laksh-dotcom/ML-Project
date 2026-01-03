import sys
import numpy as np
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformtionConfig:
    preprocessing_obj_file_path = os.path.join("artifacts", "preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformtionConfig()

    def get_data_transformer_object(self):
        try:
            num_features = [
                'writing_score', 
                'reading_score'
                ]
            
            cat_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = 'median')),
                    ("Standard Scaling", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("One_hot_encoder", OneHotEncoder())
                ]
            )
            logging.info("Numerical and Categorical Pipelines created.")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test Data sets.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            num_columns = [
                'writing_score', 'reading_score'
                ]
            
            input_train_df = train_df.drop(columns = [target_column_name], axis = 1)

        except:
            pass
        
