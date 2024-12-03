import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection as train_test_split
from dataclasses import dataclass

# Wherever we are perforiming the data injection componment there should be some inputs required by the data injection 
# componment, the input can be like where we have to probably save the trianing data and where to save the test data 
# and we are to save the raw data.
#  - So this type of input we basically be creating in another class and this type of class is this type of class is called 
# the data injection class.

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # os.path.join - here we basically creating the path where our output data will be stored which is under the artificate folder under train.csv.
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifact',"data.csv")

