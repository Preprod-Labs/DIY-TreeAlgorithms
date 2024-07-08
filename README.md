# DIY-SupervisedClassifiers

This is the Tree Algorithms Branch.

# Tree Algorithm Classifiers

|                Random Forest            |              Extra Trees         |                
|-------------------------------------|-------------------------------------|
Random Forest is an ensemble learning method for classification and regression. It constructs multiple decision trees during training and outputs the majority class (classification) or average prediction (regression). By averaging the results of trees trained on different data subsets and features, it reduces overfitting and improves generalization. This method is robust to noise and outliers and provides a measure of feature importance.          |Extra Trees (Extremely Randomized Trees) is an ensemble learning technique similar to Random Forest. It builds multiple unpruned decision trees using the entire training sample and selects the best split from random candidate splits. This extra randomization reduces variance and often improves predictive accuracy. Extra Trees is computationally efficient and handles a large number of features well.  

## Problem Definition

Develop a microservices-based architecture for Random Forest and Extra Trees Supervised Classifiers to predict customer purchases using a dataset with features such as age, annual income, credit score, and gender to predict purchase status.

## Data Definition

Mock data for learning purposes with features: Customer ID, Age, Annual Income, Credit Score, Gender, and Purchase status.

> **Note:** The dataset consists of 1000 samples, leading to potential overfitting with a high training accuracy. This would not occur in real-life scenarios with larger and more varied datasets, providing a more realistic accuracy.

## Directory Structure

- **Code/**: Contains all the scripts for data ingestion, transformation, loading, evaluation, model training, inference, manual prediction, and API.
- **Data/**: Contains the raw data and processed database.

## Data Splitting

- **Training Samples**: 600
- **Testing Samples**: 150
- **Validation Samples**: 150
- **Supervalidation Samples**: 100

## Program Flow

1. **Data Ingestion and Transformation:** Extract data from 'Data/Master', transform it, and store it in a SQLite database. [`ingest_transform.py`]
2. **Data Loading:** Load transformed data from the SQLite database. [`load.py`]
3. **Evaluation:** This code has the function to evaluate the performance of trained models on a given dataset. [`evalaute.py`]
4. **Model Training:** Train a Random Forest model [`train_randomforest.py`] and an Extra Trees model [`train_extratrees.py`] using the training data.
5. **Model Evaluation:** Evaluate the models on testing, validation, and supervalidation datasets, and generate classification reports. [`model_inference.py`]
6. **Manual Prediction:** Predict purchase status based on user input data. [`manual_prediction.py`]
7. **Web Application:** Streamlit app to provide a user-friendly GUI for predictions. [`app.py`]

## Steps to Run

1. Install the necessary packages: `pip install -r requirements.txt`
2. Run the Streamlit web application: `app.py`
