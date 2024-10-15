import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        logging.info(f"Data ingestion initialized with directory: {data_dir}")

    def load_data(self, filename):
        """Load data from a CSV file and modify column names."""
        try:
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File {filename} not found in directory {self.data_dir}.")
            
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully from {filename}.")

            # Modify column names
            df.columns = df.columns.str.replace('-', '')  
            df.columns = df.columns.str.replace('%', '1')  
            df.columns = df.columns.str.replace('#', '') 
            df.columns = df.columns.str.replace('.', '') 

            logging.info("Column names modified successfully.")
            return df
        
        except FileNotFoundError as fnf_error:
            logging.error(fnf_error)
            raise CustomException("Data file not found.") from fnf_error
        except pd.errors.EmptyDataError:
            logging.error("No data found in the file.")
            raise CustomException("The data file is empty.")
        except pd.errors.ParserError:
            logging.error("Error parsing the data file.")
            raise CustomException("Error parsing the data file.")
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise CustomException("Failed to load data.") from e

    def validate_data(self, df, required_columns):
        """Validate that the DataFrame contains the required columns."""
        try:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in the DataFrame: {missing_columns}")
            
            logging.info("Data validation successful. All required columns are present.")
        
        except ValueError as ve:
            logging.error(ve)
            raise CustomException("Data validation failed.") from ve
        except Exception as e:
            logging.error(f"An error occurred during data validation: {e}")
            raise CustomException("Data validation error.") from e


if __name__ == "__main__":
    data_directory = 'D:/Data_science/Healthcare_project/OVARIAN_CANCER_PREDICTION/data/'  
    ingestion = DataIngestion(data_directory)

    try:
        
        df_premenopausal = ingestion.load_data('data_pre_meno.csv')
        df_postmenopausal = ingestion.load_data('data_post_meno.csv')
        
        required_columns = ['AFP', 'AG', 'Age', 'ALB', 'ALP', 'ALT', 'AST', 'BASO', 'BASO1', 'BUN',
       'Ca', 'CA125', 'CA199', 'CA724', 'CEA', 'CL', 'CO2CP', 'CREA', 'DBIL',
       'EO', 'EO1', 'GGT', 'GLO', 'GLU', 'HCT', 'HE4', 'HGB', 'IBIL', 'K',
       'LYM', 'LYM1', 'MCH', 'MCV', 'Mg', 'MONO', 'MONO1', 'MPV', 'Na', 'NEU',
       'PCT', 'PDW', 'PHOS', 'PLT', 'RBC', 'RDW', 'TBIL', 'TP', 'UA', 'TYPE']

        ingestion.validate_data(df_premenopausal, required_columns)
        ingestion.validate_data(df_postmenopausal, required_columns)

    except CustomException as ce:
        logging.error(f"Data ingestion error: {ce}")
