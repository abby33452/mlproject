import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import traceback

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')

            # Print the full paths for debugging
            print(f"Model path: {os.path.abspath(model_path)}")
            print(f"Preprocessor path: {os.path.abspath(preprocessor_path)}")

            # Check if the files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No such file or directory: '{os.path.abspath(model_path)}'")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"No such file or directory: '{os.path.abspath(preprocessor_path)}'")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            print(f"An error occurred in predict: {e}")
            traceback.print_exc()
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            print(f"An error occurred in get_data_as_data_frame: {e}")
            traceback.print_exc()
            raise CustomException(e, sys)