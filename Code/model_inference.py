# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Mohini T and Vansh R
        # Role: Architects
        # Code ownership rights: Mohini T and Vansh R
    # Version:
        # Version: V 1.1 (04 July 2024)
            # Developer: Mohini T and Vansh R
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code performs inference using the trained model and evaluates it on testing, validation, and supervalidation datasets.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Pandas 2.2.2
        # Scikit-learn 1.5.0

import pandas as pd # For data manipulation and analysis
import joblib # For loading the trained model

# Importing helper functions from the local .py files
from evaluate import evaluate_model
from load import load_eval

def model_inference(db_path, model_path):
    
    # Load the evaluation data
    test_data, val_data, superval_data = load_eval(db_path)
    
    # Load the model
    model = joblib.load(model_path)
    
    # Evaluate the model on the testing, validation, and supervalidation datasets
    test_accuracy, test_report, test_cross_val_scores = evaluate_model(model, test_data)
    val_accuracy, val_report, val_cross_val_scores = evaluate_model(model, val_data)
    superval_accuracy, superval_report, superval_cross_val_scores = evaluate_model(model, superval_data)
    
    return test_accuracy, test_report, test_cross_val_scores, val_accuracy, val_report, val_cross_val_scores, superval_accuracy, superval_report, superval_cross_val_scores