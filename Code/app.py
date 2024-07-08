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
     
    # Description: This Streamlit app allows users to input features and make predictions using a pre-trained model.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0

import streamlit as st # For building the web app

# Importing helper functions from the local .py files
from manual_prediction import manual_prediction
from train_randomforest import train_model as train_random_forest_model
from train_extratrees import train_model as train_extra_trees_model
from model_inference import model_inference

st.set_page_config(page_title="Customer Purchase Prediction", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Customer Price Prediction </h1>", unsafe_allow_html=True)
st.divider()

# Declaring session states(streamlit variables) for saving the path throught page reloads
    
# This is how we declare session state variables in streamlit.
if "db_path" not in st.session_state:
    st.session_state.db_path = "Data/Processed/sqlite_db.db"
    
if "raw_data_path" not in st.session_state:
    st.session_state.raw_data_path = "Data/Master/mock_data.csv"
    
if "random_forest_model_path" not in st.session_state:
    st.session_state.random_forest_model_path = "random_forest.pkl"
    
if "extra_trees_model_path" not in st.session_state:
    st.session_state.extra_trees_model_path = "extra_trees.pkl"

tab1, tab2, tab3, tab4 =  st.tabs(["Model Config", "Model Training", "Model Evaluation", "Model Prediction"])

# Tab for Model Config
with tab1:
    st.subheader("Model Config")
    st.write("This is where you can set your paths for the model.")
    st.divider()
        
    # Taking the path input from the user:
    
    with st.form(key="model_config_form"):
        db_path = st.text_input("Enter the path to the SQLite database:", 
                                value=st.session_state.db_path)
        st.session_state.db_path = db_path
        
        raw_data_path = st.text_input("Enter the path to the raw data:", 
                                      value=st.session_state.raw_data_path)
        st.session_state.raw_data_path = raw_data_path
        
        random_forest_model_path = st.text_input("Enter the path to save the Random Forest model:",
                                                 value=st.session_state.random_forest_model_path)
        st.session_state.random_forest_model_path = random_forest_model_path
        
        extra_trees_model_path = st.text_input("Enter the path to save the Extra Trees model:", 
                                               value=st.session_state.extra_trees_model_path)
        st.session_state.extra_tress_model_path = extra_trees_model_path
        
        if st.form_submit_button("Save"):
            st.success("Paths saved successfully.")

# Tab for Model Training
with tab2:
    st.subheader("Model Training")
    st.write("This is where you can train the model.")
    st.divider()
    
    st.markdown("<h3 style='text-align: center; color: white;'>Random Forest Model </h3>", unsafe_allow_html=True)
    if st.button("Train Random Forest Model", use_container_width=True):
        
        with st.status("Training Random Forest Model..."):
            rf_accuracy, rf_report, rf_cross_val_scores =  train_random_forest_model(st.session_state.db_path, st.session_state.random_forest_model_path)
        
        st.success("Random Forest model trained successfully.")
        
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)
        
        # Print the performance metrics
        st.write(f"Accuracy: {rf_accuracy}")
        st.text(f"Classification Report: {rf_report}")
        st.write(f"Cross Validation Scores: {rf_cross_val_scores}")
        
    st.divider()
    
    st.markdown("<h3 style='text-align: center; color: white;'>Extra Trees Model </h3>", unsafe_allow_html=True)
    if st.button("Train Extra Trees Model", use_container_width=True):
        
        with st.status("Training Extra Trees Model..."):
            et_accuracy, et_report, et_cross_val_cores = train_extra_trees_model(st.session_state.db_path, st.session_state.extra_trees_model_path)
        
        st.success("Extra Trees model trained successfully.")
        
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)
        
        # Print the performance metrics
        st.write(f"Accuracy: {et_accuracy}")
        st.text(f"Classification Report: {et_report}")
        st.write(f"Cross Validation Scores: {et_cross_val_cores}")
        
        
    st.divider()
    
# Tab for Model Evaluation
with tab3:
    st.subheader("Model Evaluation")
    st.write("This is where you can see the current metrics of the latest saved model.")
    st.divider()
    
    st.markdown("<h3 style='text-align: center; color: white;'>Random Forest Model </h3>", unsafe_allow_html=True)
    rf_test_accuracy, rf_test_report, rf_test_cross_val_scores, rf_val_accuracy, rf_val_report, rf_val_cross_val_scores, rf_superval_accuracy, rf_superval_report, rf_superval_cross_val_scores = model_inference(st.session_state.db_path, st.session_state.random_forest_model_path)
    
    # Test Data Metrics
    st.markdown("<h4 style='text-align: center; color: white;'>Testing Data </h4>", unsafe_allow_html=True)
    st.write(f"Accuracy: {rf_test_accuracy}")
    st.text(f"Classification Report: {rf_test_report}")
    st.write(f"Cross Validation Scores: {rf_test_cross_val_scores}")
    
    # Validation Data Metrics
    st.markdown("<h4 style='text-align: center; color: white;'>Validation Data </h4>", unsafe_allow_html=True)
    st.write(f"Accuracy: {rf_val_accuracy}")
    st.text(f"Classification Report: {rf_val_report}")
    st.write(f"Cross Validation Scores: {rf_val_cross_val_scores}")
    
    # Supervalidation Data Metrics
    st.markdown("<h4 style='text-align: center; color: white;'>Supervalidation Data </h4>", unsafe_allow_html=True)
    st.write(f"Accuracy: {rf_superval_accuracy}")
    st.text(f"Classification Report: {rf_superval_report}")
    st.write(f"Cross Validation Scores: {rf_superval_cross_val_scores}")
    
    st.divider()
    
    st.markdown("<h3 style='text-align: center; color: white;'>Extra Trees Model </h3>", unsafe_allow_html=True)
    et_test_accuracy, et_test_report, et_test_cross_val_scores, et_val_accuracy, et_val_report, et_val_cross_val_scores, et_superval_accuracy, et_superval_report, et_superval_cross_val_scores = model_inference(st.session_state.db_path, st.session_state.extra_trees_model_path)
    
    # Test Data Metrics
    st.markdown("<h4 style='text-align: center; color: white;'>Testing Data </h4>", unsafe_allow_html=True)
    st.write(f"Accuracy: {et_test_accuracy}")
    st.text(f"Classification Report: {et_test_report}")
    st.write(f"Cross Validation Scores: {et_test_cross_val_scores}")
    
    # Validation Data Metrics
    st.markdown("<h4 style='text-align: center; color: white;'>Validation Data </h4>", unsafe_allow_html=True)
    st.write(f"Accuracy: {et_val_accuracy}")
    st.text(f"Classification Report: {et_val_report}")
    st.write(f"Cross Validation Scores: {et_val_cross_val_scores}")
    
    # Supervalidation Data Metrics
    st.markdown("<h4 style='text-align: center; color: white;'>Supervalidation Data </h4>", unsafe_allow_html=True)
    st.write(f"Accuracy: {et_superval_accuracy}")
    st.text(f"Classification Report: {et_superval_report}")
    st.write(f"Cross Validation Scores: {et_superval_cross_val_scores}")
    
    
    st.divider()

# Tab for Model Prediction    
with tab4:
    st.subheader("Model Prediction")
    st.write("Enter the customer details to predict if they will make a purchase.")
    st.divider()

    with st.form(key="model_prediction_form"):
        
        model = st.selectbox("Select model:", ("Random Forest", "Extra Trees"))
        age = st.number_input('Enter age (18-80):', min_value=18, max_value=80, value=30)
        annual_income = st.number_input('Enter annual income (30000.00-100000.00):', min_value=30000.00, max_value=100000.00, value=50000.00, step=0.01, format="%.2f")
        credit_score = st.number_input('Enter credit score (300-850):', min_value=300, max_value=850, value=600)
        gender = st.selectbox('Select gender:', ('Male', 'Female'))
        
        if model == "Random Forest":
            model_path = st.session_state.random_forest_model_path
        else:
            model_path = st.session_state.extra_trees_model_path
            
        if st.form_submit_button("Predict", use_container_width=True):
            prediction = manual_prediction(model_path, age, annual_income, credit_score, gender)
            st.write(f"Prediction: {prediction}")