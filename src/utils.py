import os
import pandas as pd
from src.exception import CustomException
import sys

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data in the form of a DataFrame.
    """
    try:
        if not os.path.exists(file_path):
            raise CustomException(f"File {file_path} does not exist.", sys)
        
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise CustomException(f"Error occurred while loading the data from {file_path}", sys)
