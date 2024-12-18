import os
import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#To create Class variable of data components

@dataclass
class DataIngestionConfig:
# below are the INPUT giving to the Data ingestion component
# and data ingestion take and store the files in the below mentioned directory / folder
# with specific name
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
# all Outputs will be saved in the artifacts folder, which are train, test and raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the data from different sources
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            
            # make directory if not exist 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            #Saving data in the raw datapath
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            # Return values are important for Data transformation
            # Train and Test data point 
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except FileNotFoundError as fnf_e:
            logging.error(f"FileNotFoundError: {fnf_e}")
            raise CustomException(fnf_e, sys)
        except Exception as e:
            logging.error(f"Exception: {e}")
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))