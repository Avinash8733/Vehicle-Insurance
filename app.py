import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# Set page config
st.set_page_config(page_title="Vehicle Insurance Fraud Detection", layout="wide")

# Embedded Schema
DATA_SCHEMA = {
    "Month": ["Apr", "Aug", "Dec", "Feb", "Jan", "Jul", "Jun", "Mar", "May", "Nov", "Oct", "Sep"],
    "WeekOfMonth": [1, 2, 3, 4, 5],
    "DayOfWeek": ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"],
    "Make": ["Accura", "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda", "Jaguar", "Lexus", "Mazda", "Mecedes", "Mercury", "Nisson", "Pontiac", "Porche", "Saab", "Saturn", "Toyota", "VW"],
    "AccidentArea": ["Rural", "Urban"],
    "DayOfWeekClaimed": ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"],
    "MonthClaimed": ["Apr", "Aug", "Dec", "Feb", "Jan", "Jul", "Jun", "Mar", "May", "Nov", "Oct", "Sep"],
    "WeekOfMonthClaimed": [1, 2, 3, 4, 5],
    "Sex": ["Female", "Male"],
    "MaritalStatus": ["Divorced", "Married", "Single", "Widow"],
    "Age": {"type": "number", "min": 0.0, "max": 80.0, "median": 38.0},
    "Fault": ["Policy Holder", "Third Party"],
    "PolicyType": ["Sedan - All Perils", "Sedan - Collision", "Sedan - Liability", "Sport - All Perils", "Sport - Collision", "Sport - Liability", "Utility - All Perils", "Utility - Collision", "Utility - Liability"],
    "VehicleCategory": ["Sedan", "Sport", "Utility"],
    "VehiclePrice": ["20000 to 29000", "30000 to 39000", "40000 to 59000", "60000 to 69000", "less than 20000", "more than 69000"],
    "RepNumber": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "Deductible": [300, 400, 500, 700],
    "DriverRating": [1, 2, 3, 4],
    "Days_Policy_Accident": ["1 to 7", "15 to 30", "8 to 15", "more than 30", "none"],
    "Days_Policy_Claim": ["15 to 30", "8 to 15", "more than 30"],
    "PastNumberOfClaims": ["1", "2 to 4", "more than 4", "none"],
    "AgeOfVehicle": ["2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7", "new"],
    "AgeOfPolicyHolder": ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65"],
    "PoliceReportFiled": ["No", "Yes"],
    "WitnessPresent": ["No", "Yes"],
    "AgentType": ["External", "Internal"],
    "NumberOfSuppliments": ["1 to 2", "3 to 5", "more than 5", "none"],
    "AddressChange_Claim": ["1 year", "2 to 3 years", "4 to 8 years", "no change", "under 6 months"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
    "Year": [1994, 1995, 1996],
    "BasePolicy": ["All Perils", "Collision", "Liability"]
}

def train_model():
    """Retrains the model if pickle is missing or incompatible."""
    try:
        if not os.path.exists('fraud_oracle.csv'):
            st.error("Dataset 'fraud_oracle.csv' not found. Cannot retrain model.")
            return None
        
        df = pd.read_csv('fraud_oracle.csv')
        # Cleaning
        df = df[df['DayOfWeekClaimed'] != '0']
        df = df[df['MonthClaimed'] != '0']
        if 'PolicyNumber' in df.columns:
            df = df.drop('PolicyNumber', axis=1)
        
        target_col = 'FraudFound_P'
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Train
        pipeline.fit(X, y)
        
        # Save
        joblib.dump(pipeline, 'pipeline.pkl')
        return pipeline
        
    except Exception as e:
        st.error(f"Training failed: {e}")
        return None

@st.cache_resource
def load_pipeline():
    # Try loading existing pipeline
    pipeline = None
    if os.path.exists('pipeline.pkl'):
        try:
            pipeline = joblib.load('pipeline.pkl')
        except Exception:
            pipeline = None # Force retrain if load fails
            
    # If no pipeline or load failed, retrain
    if pipeline is None:
        pipeline = train_model()
        
    return pipeline

def main():
    st.title("🚗 Vehicle Insurance Claim Fraud Detection")
    st.markdown("""
    This application predicts the likelihood of a vehicle insurance claim being **Fraudulent**.
    Please enter the claim details in the sidebar.
    """)

    # Load Pipeline
    with st.spinner("Loading/Training Model..."):
        pipeline = load_pipeline()
    
    if pipeline is None:
        st.error("Failed to load or train model.")
        return

    # Sidebar for inputs
    st.sidebar.header("Claim Details")
    
    input_data = {}

    # Iterate over schema to create inputs
    for col, values in DATA_SCHEMA.items():
        if isinstance(values, list):
            options = values
            input_data[col] = st.sidebar.selectbox(f"{col}", options)
        elif isinstance(values, dict) and values.get('type') == 'number':
            val = float(values['median'])
            min_v = float(values['min'])
            max_v = float(values['max'])
            input_data[col] = st.sidebar.number_input(f"{col}", min_value=min_v, max_value=max_v, value=val)

    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data])
    
    st.subheader("Prediction Result")
    
    if st.button("Predict Fraud Status"):
        try:
            # Predict
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]
            
            # Display
            if prediction == 1:
                st.error("🚨 FRAUD DETECTED")
                st.markdown(f"**Probability of Fraud:** {probability:.2%}")
                st.warning("This claim shows characteristics typically associated with fraudulent claims.")
            else:
                st.success("✅ CLAIM APPROVED")
                st.markdown(f"**Probability of Fraud:** {probability:.2%}")
                st.info("This claim appears to be legitimate.")

        except Exception as e:
            # If prediction fails (e.g. version mismatch even after load), force retrain
            st.warning(f"Error during prediction ({e}). Attempting to retrain model...")
            try:
                os.remove('pipeline.pkl') # Force delete
                st.cache_resource.clear()
                pipeline = train_model()
                if pipeline:
                    prediction = pipeline.predict(input_df)[0]
                    probability = pipeline.predict_proba(input_df)[0][1]
                    if prediction == 1:
                        st.error("🚨 FRAUD DETECTED")
                        st.markdown(f"**Probability of Fraud:** {probability:.2%}")
                    else:
                        st.success("✅ CLAIM APPROVED")
                        st.markdown(f"**Probability of Fraud:** {probability:.2%}")
                else:
                    st.error("Retraining failed.")
            except Exception as e2:
                st.error(f"Critical error: {e2}")

if __name__ == "__main__":
    main()
