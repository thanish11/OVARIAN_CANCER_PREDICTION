import pandas as pd
import xgboost as xgb
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
import sys

# GA-based feature selection function using GeneticSelectionCV
def ga_feature_selection(X, y):
    try:
        # Initialize XGBoost classifier
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        # Apply GeneticSelectionCV for feature selection
        selector = GeneticSelectionCV(
            estimator=model,
            cv=5,
            verbose=1,
            scoring="accuracy",
            max_features=X.shape[1],  # Allow the GA to consider all features
            n_population=200,  # Number of individuals in the population
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=40,  # Number of generations to run
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            n_gen_no_change=10,  # Stop if no improvement over 10 generations
            n_jobs=-1  # Use all available cores for parallel processing
        )

        # Fit the selector
        selector = selector.fit(X, y)
        
        # Get the selected features
        selected_features = X.columns[selector.support_]
        
        logging.info(f"Selected features: {selected_features}")
        return selected_features

    except Exception as e:
        logging.error(f"Error in GA feature selection: {e}")
        raise CustomException(e, sys)

# Preprocess datasets
def preprocess_datasets(data_path, label_column):
    """Preprocess the dataset: remove the target column and split data."""
    try:
        data = pd.read_csv(data_path)
        X = data.drop(columns=[label_column])
        y = data[label_column]
        return X, y
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise CustomException(e, sys)

# Perform feature selection for both premenopausal and postmenopausal data
if __name__ == "__main__":
    try:
        # Paths to datasets
        premeno_data_path = "data/premenopausal_data.csv"
        postmeno_data_path = "data/postmenopausal_data.csv"
        
        # Column containing the target labels
        label_column = "TYPE"  # Replace with the actual label column in your dataset

        # Preprocess premenopausal data
        logging.info("Processing premenopausal dataset...")
        X_premeno, y_premeno = preprocess_datasets(premeno_data_path, label_column)
        selected_features_premeno = ga_feature_selection(X_premeno, y_premeno)
        logging.info(f"Selected features for premenopausal: {selected_features_premeno}")

        # Preprocess postmenopausal data
        logging.info("Processing postmenopausal dataset...")
        X_postmeno, y_postmeno = preprocess_datasets(postmeno_data_path, label_column)
        selected_features_postmeno = ga_feature_selection(X_postmeno, y_postmeno)
        logging.info(f"Selected features for postmenopausal: {selected_features_postmeno}")

        # Output selected features
        print(f"Selected Features for Premenopausal: {selected_features_premeno}")
        print(f"Selected Features for Postmenopausal: {selected_features_postmeno}")

    except CustomException as ce:
        logging.error(f"Feature selection failed: {ce}")
