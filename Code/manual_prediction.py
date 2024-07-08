# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Mohini T and Vansh R
        # Role: Architects
        # Code ownership rights: Mohini T and Vansh R
    # Version:
        # Version: V 1.1 (04 July 2024)
            # Developers: Mohini T and Vansh R
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code takes user inputs for features and makes predictions using a pre-trained model.
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
        # Joblib 1.4.2

import pandas as pd # For data manipulation and analysis
import joblib # For loading the pre-trained model

def prepare_input(age, annual_income, credit_score, gender):
    # Convert gender to numerical
    gender = 0 if gender == 'male' else 1 # 0 is Male and 1 is Female
    
    # Create a dictionary with user input data
    user_data = {
        'age': age,
        'annual_income': round(annual_income, 2),
        'credit_score': credit_score,
        'gender': gender
    }

    # Convert the dictionary to a DataFrame
    features = pd.DataFrame([user_data])
    return features

def predict(model, user_input):
    # Make a prediction using the model and user input
    prediction = model.predict(user_input)[0]
    return 'Purchase' if prediction else 'No Purchase'

def manual_prediction(model_path, age, annual_income, credit_score, gender):
    
    # Load the model
    model = joblib.load(model_path)
    
    # Prepare user input data
    user_data = prepare_input(age, annual_income, credit_score, gender)
    
    if user_data is not None:
        # Make a prediction using the model and user data
        prediction = predict(model, user_data)  
        
        # Print the prediction result
        return prediction