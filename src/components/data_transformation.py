import sys
import os
 
import numpy as np
import pandas as pd 
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object


@dataclass
class DataTransformationConfig :
    preprocessor_objet_file_path = os.path.join('Entire_data','preprocessor.pkl')
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def data_transformer_object(self):
        try:
            num_columns = ['reading score', 'writing score']
            cat_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            num_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())  
                ]
            )
            
            cat_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("One_Hot_Encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean = False))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [("num_pipline", num_pipline,num_columns),
                ("cat_pipline", cat_pipline,cat_columns)]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformer(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessing_obj=self.data_transformer_object()

            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis = 1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_objet_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_objet_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
    
    




 