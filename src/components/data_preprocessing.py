from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys


class DataPreprocessing:
    def __init__(self):
        logging.info("Data Preprocessing initialized.")

    def mice_imputation(self, df):
        """Perform MICE imputation for missing data."""
        try:
            imputer = IterativeImputer()  # MICE imputation
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            logging.info("MICE Imputation completed.")
            return df_imputed
        except Exception as e:
            logging.error("Error during MICE imputation.")
            raise CustomException(e, sys)

    def detect_outliers_iqr(self, df):
        """Detect and handle outliers using the IQR method."""
        try:
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df_outliers_removed = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
            logging.info("Outliers handled using IQR method.")
            return df_outliers_removed
        except Exception as e:
            logging.error("Error during outlier detection.")
            raise CustomException(e, sys)

    def data_normalization(self, df):
        """Normalize data using StandardScaler."""
        try:
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            logging.info("Data normalization completed.")
            return df_scaled
        except Exception as e:
            logging.error("Error during data normalization.")
            raise CustomException(e, sys)

    def preprocess_data(self, df):
        """Execute the full preprocessing pipeline."""
        try:
            logging.info("Starting full preprocessing pipeline.")

            # Step 1: MICE Imputation for missing data
            df_imputed = self.mice_imputation(df)

            # Step 2: Detect and handle outliers using IQR
            df_clean = self.detect_outliers_iqr(df_imputed)

            # Step 3: Normalize the data
            df_normalized = self.data_normalization(df_clean)

            logging.info("Preprocessing pipeline completed successfully.")
            return df_normalized

        except Exception as e:
            logging.error("Error during preprocessing pipeline.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Load your dataset (replace with actual paths for pre- and postmenopausal data)
        df_premeno = pd.read_csv('data/data_pre_meno.csv')
        df_postmeno = pd.read_csv('data/data_post_meno.csv')

        # Initialize preprocessing for both datasets
        preprocessor = DataPreprocessing()

        # Preprocess premenopausal data
        logging.info("Preprocessing premenopausal data.")
        df_premeno_processed = preprocessor.preprocess_data(df_premeno)

        # Preprocess postmenopausal data
        logging.info("Preprocessing postmenopausal data.")
        df_postmeno_processed = preprocessor.preprocess_data(df_postmeno)

        # Save preprocessed data for further steps
        df_premeno_processed.to_csv('data/premenopausal_data_processed.csv', index=False)
        df_postmeno_processed.to_csv('data/postmenopausal_data_processed.csv', index=False)

    except CustomException as ce:
        logging.error(f"An error occurred during preprocessing: {ce}")
