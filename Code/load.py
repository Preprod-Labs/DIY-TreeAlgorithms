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
     
    # Description: This code loads data from sqlite_db.db and saves it into .csv files in the Processed folder.
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

import pandas as pd # For data manipulation and analysis
import sqlite3 # For connecting to and interacting with SQLite databases

def load_from_sql(db_path, table_name):
    # Connects to sqlite db and returns the data in the form of a dataframe
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Load the data from the database
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    
    # Close the connection
    conn.close()
    
    return df

def load_train(db_path):
    
    # Load the data from SQL database
    train_data = load_from_sql(db_path, "training_data")
    
    return train_data
    
    
def load_eval(db_path):
    test_data = load_from_sql(db_path, "testing_data")
    val_data = load_from_sql(db_path, "validation_data")
    superval_data = load_from_sql(db_path, "supervalidation_data")
    
    return test_data, val_data, superval_data
    