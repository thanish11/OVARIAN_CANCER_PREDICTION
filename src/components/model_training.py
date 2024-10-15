import os
import sys
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import load_data

class ModelTraining:
    def __init__(self, preprocessed_premenopausal_file, preprocessed_postmenopausal_file, model_save_dir):
        self.premenopausal_file = preprocessed_premenopausal_file
        self.postmenopausal_file = preprocessed_postmenopausal_file
        self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

    def train_xgboost(self, X_train, y_train):
        try:
            logging.info("Training XGBoost model...")
            model = xgb.XGBClassifier(eval_metric='logloss')
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error during XGBoost training: {e}")
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            logging.info(f"Accuracy: {acc}")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"AUC-ROC Score: {roc_auc}")
            logging.info(f"Classification Report: \n{classification_report(y_test, y_pred)}")

            return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise CustomException(e, sys)

    def save_model(self, model, model_name):
        try:
            model_path = os.path.join(self.model_save_dir, model_name)
            joblib.dump(model, model_path)
            logging.info(f"Model saved to {model_path}")
        except Exception as e:
            logging.error(f"Error while saving model: {e}")
            raise CustomException(e, sys)

    def run(self):
        try:
            # Load preprocessed datasets
            logging.info("Loading preprocessed datasets...")
            df_premenopausal = pd.read_csv(self.premenopausal_file)
            df_postmenopausal = pd.read_csv(self.postmenopausal_file)

            # Splitting features and labels
            X_premenopausal = df_premenopausal.drop(columns=['TYPE'])  # Assuming 'TYPE' is the target column
            y_premenopausal = df_premenopausal['TYPE']

            X_postmenopausal = df_postmenopausal.drop(columns=['TYPE'])
            y_postmenopausal = df_postmenopausal['TYPE']

            # Train XGBoost models for premenopausal and postmenopausal datasets
            logging.info("Training XGBoost model for premenopausal dataset...")
            premeno_model = self.train_xgboost(X_premenopausal, y_premenopausal)
            self.save_model(premeno_model, "xgboost_premenopausal_model.pkl")

            logging.info("Training XGBoost model for postmenopausal dataset...")
            postmeno_model = self.train_xgboost(X_postmenopausal, y_postmenopausal)
            self.save_model(postmeno_model, "xgboost_postmenopausal_model.pkl")

            # Evaluate both models
            logging.info("Evaluating premenopausal model...")
            self.evaluate_model(premeno_model, X_premenopausal, y_premenopausal)

            logging.info("Evaluating postmenopausal model...")
            self.evaluate_model(postmeno_model, X_postmenopausal, y_postmenopausal)

        except Exception as e:
            logging.error(f"Error in model training pipeline: {e}")
            raise CustomException(e, sys)

# Example usage:
if __name__ == "__main__":
    premeno_data_file = "data/data_pre_meno.csv"
    postmeno_data_file = "data/data_post_meno.csv"
    model_save_directory = "models/"

    trainer = ModelTraining(premeno_data_file, postmeno_data_file, model_save_directory)
    trainer.run()
