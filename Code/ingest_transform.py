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
     
    # Description: This code enables data ingestion from original_data.csv, transformation, and storage into sqlite_db.db.
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
from sklearn.model_selection import train_test_split # For splitting the data

def preprocess_data(df):
    # Drop customer_id column
    df = df.drop(columns=['customer_id'])
    
    # Convert gender to numerical values
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    return df

def split_data(df):
    # Shuffle the data before splitting
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split data into training (600) and temp (400)
    train_data, temp_data = train_test_split(df, train_size=600, random_state=42)
    # Split temp data into testing (150) and remaining (250)
    test_data, val_superval_data = train_test_split(temp_data, train_size=150, random_state=42)
    # Split remaining data into validation (150) and supervalidation (100)
    val_data, superval_data = train_test_split(val_superval_data, train_size=150, random_state=42)
    
    return train_data, test_data, val_data, superval_data

def save_to_sql(train_data, test_data, val_data, superval_data, db_path):
    # Create a new SQLite database
    conn = sqlite3.connect(db_path)
    
    # Save the data to the database
    train_data.to_sql('training_data', conn, index=False, if_exists='replace')
    test_data.to_sql('testing_data', conn, index=False, if_exists='replace')
    val_data.to_sql('validation_data', conn, index=False, if_exists='replace')
    superval_data.to_sql('supervalidation_data', conn, index=False, if_exists='replace')
    
    # Close the connection
    conn.close()

def display_sample_counts(train_data, test_data, val_data, superval_data):
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Supervalidation samples: {len(superval_data)}")

def ingest_transform(raw_data_path, db_path):
    
    # Read the raw data
    df = pd.read_csv(raw_data_path)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Split the data into training, testing, validation, and supervalidation sets
    train_data, test_data, val_data, superval_data = split_data(df)
    
    # Display the sample counts for each set
    display_sample_counts(train_data, test_data, val_data, superval_data)
    
    # Save the split data to SQL database
    save_to_sql(train_data, test_data, val_data, superval_data, db_path)
    
    # Print a success message
    print("Data preprocessing, splitting, and saving to SQL database completed successfully.")
    
if __name__ == "__main__":
    ingest_transform('Data/Master/mock_data.csv', 'Data/Processed/sqlite_db.db')