import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from catboost import CatBoostRegressor
from src.exception import CustomException


def save_object(file_path, obj):
    """Save an object using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load an object from a pickle file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluates multiple models and returns their R² scores."""
    try:
        report = {}

        for model_name, model in models.items():
            print(f"\n🔍 Evaluating Model: {model_name}, Type: {type(model)}")

            # Ensure model is instantiated
            if isinstance(model, type):
                model = model()

            # Special handling for CatBoost
            if isinstance(model, CatBoostRegressor):
                model.fit(X_train, y_train, verbose=False)
                best_model = model
                best_params = "Default CatBoost Params"
            else:
                # Check if model is a valid scikit-learn estimator
                if not isinstance(model, BaseEstimator):
                    print(f"⚠️ Skipping {model_name}: Not a scikit-learn estimator.")
                    continue
                
                param_grid = param.get(model_name, {})

                # If param_grid is empty, fit model directly
                if param_grid:
                    print(f"📌 Hyperparameter Grid: {param_grid}")
                    gs = GridSearchCV(model, param_grid, cv=3)
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                    best_params = gs.best_params_
                else:
                    print(f"⚠️ No hyperparameter tuning for {model_name}. Training default model.")
                    model.fit(X_train, y_train)  # Fit model manually if no parameters
                    best_model = model
                    best_params = "Default Parameters"

            print(f"✅ Best Parameters for {model_name}: {best_params}")

            # Ensure the model is fitted before prediction
            if not hasattr(best_model, "predict"):
                print(f"❌ {model_name} is not fitted properly. Skipping...")
                continue

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

            print(f"📊 {model_name} Test R² Score: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
