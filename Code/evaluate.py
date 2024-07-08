# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Mohini T and Vansh R
        # Role: Architects
        # Code ownership rights: Mohini T and Vansh R
    # Version:
        # Version: V 1.0 (04 July 2024)
            # Developer: Mohini T and Vansh R
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code evaluates the performance of a trained model on a given dataset.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Scikit-learn 1.5.0

from sklearn.metrics import accuracy_score, classification_report # For evaluating the model
from sklearn.model_selection import cross_val_score # For cross-validation

def evaluate_model(model, data):
    # Prepare the input features
    X = data.drop(columns=['purchase'])
    
    # Prepare the true labels
    y_true = data['purchase']
    
    # Predict the labels using the trained model
    y_pred = model.predict(X)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cross_val_scores = cross_val_score(model, X, y_pred, cv=5)
    
    return accuracy, report, cross_val_scores