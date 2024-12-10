import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# Wherever we are perforiming the data injection componment there should be some inputs required by the data injection 
# componment, the input can be like where we have to probably save the trianing data and where to save the test data 
# and we are to save the raw data.
#  - So this type of input we basically be creating in another class and this type of class is this type of class is called 
# the data injection class.

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # os.path.join - here we basically creating the path where our output data will be stored which is under the artificate folder under train.csv.
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

# If we are only decleration variables inside a class then we should use the dataclass but if we have other function inside the class then its a suggestion to not use the dataclass
class DataIngestion: 
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component") 
        try:
            df=pd.read_csv('notebook/data/stud.csv') # Reading the data frome the csv file.
            logging.info("Read the dataset as dataframe.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # Creating a path or folder or the location where we have to store the raw, train and test data.

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # We are exporting the original or raw data in the csv format in a particular location naming it as a 'raw_data_path'.

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42) # We are splitting the raw data into to train data and test data.

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # We are exproting the train data into a particular location naming it as 'train_data_path'.

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # We are exporting the test data into a particular location naming it as 'test_data_path'.

            logging.info("Ingestion of the data incompleted.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr, test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modelTrainer=ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))