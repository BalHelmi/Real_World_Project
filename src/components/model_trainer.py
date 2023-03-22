import os 
import sys

import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.utils import evaluate_models
from src.utils import save_object

@dataclass
class ModelTrainerConfig():
    def __init__(self):
        self.trained_model_path = os.path.join('Entire_data','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, X_train_array, Y_train_array):
        try:
            X_train,y_train,X_test,y_test=(
                X_train_array[:,:-1],
                X_train_array[:,-1],
                Y_train_array[:,:-1],
                Y_train_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
             
            # find the best score model 
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            save_object(
                file_path=self.trained_model_config.trained_model_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)

            score = r2_score(y_test, predicted)
            return score
        
        except Exception as e :
            raise CustomException(e,sys)
             
        
        