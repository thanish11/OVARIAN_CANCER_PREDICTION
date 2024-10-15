import pickle
import numpy as np
import pandas as pd
from src.utils import load_data
from src.exception import CustomException
import sys

class PredictionPipeline:
    def __init__(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise CustomException(f"Error loading model from {model_path}", sys)

    def predict(self, input_data):
        """
        Predict based on input data. Input should be a pandas DataFrame.
        """
        try:
            prediction = self.model.predict(input_data)
            return prediction
        except Exception as e:
            raise CustomException("Error during prediction.", sys)
