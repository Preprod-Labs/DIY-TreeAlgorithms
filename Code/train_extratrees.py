# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Vansh R
        # Role: Architect
        # Code ownership rights: Vansh R
    # Version:
        # Version: V 1.0 (04 July 2024)
            # Developer: Vansh R
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code trains an Extra Trees classifier on the training data and saves the model.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environmen:     
        # Python 3.11.5
        # Pandas 2.2.2
        # Joblib 1.4.2
        # Scikit-learn 1.5.0

import pandas as pd # For data manipulation and analysis
import joblib # For loading the trained model
from sklearn.ensemble import ExtraTreesClassifier # For training the model

# Importing helper functions from the local .py files
from load import load_train
from evaluate import evaluate_model

def train_model(db_path, model_path):
    
    # Read the training data from a CSV file
    train_data = load_train(db_path)

    # Prepare the data
    X_train = train_data.drop(columns=['purchase'])
    y_train = train_data['purchase']

    # Train the model
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)

    # Print a success message
    print("Extra Trees model training completed successfully.")
    
    # Return the train model metrics
    return evaluate_model(model, train_data)