import os 
import sys

import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
@dataclass
class dataIngestionConfig():
    train_data_path : str = os.path.join('Entire_data','train_data.csv')
    test_data_path : str = os.path.join('Entire_data','test_data.csv')
    data_data_path : str = os.path.join('Entire_data','data.csv')
    
class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Prearing data For Ingestion")
            df = pd.read_csv('notebook\data\student.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.data_data_path, index=False, header=False)
            train_set , test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
            

if __name__ == "__main__" :
    
    data = dataIngestion()
    train_data_path,test_data_path = data.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr, test_arr, _ =data_transformation.initiate_data_transformer(train_data_path,test_data_path)
    
    model = ModelTrainer()
    print(f"the r2_score of the model = {model.initiate_model_trainer(train_arr, test_arr)}")
    
    
    