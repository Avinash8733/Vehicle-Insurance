import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Vehicle Insurance Fraud Detection", layout="wide")

# Embedded Schema to avoid dependency on the CSV file
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

@st.cache_resource
def load_pipeline():
    return joblib.load('pipeline.pkl')

def main():
    st.title("🚗 Vehicle Insurance Claim Fraud Detection")
    st.markdown("""
    This application predicts the likelihood of a vehicle insurance claim being **Fraudulent**.
    Please enter the claim details in the sidebar.
    """)

    # Load Pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return

    # Sidebar for inputs
    st.sidebar.header("Claim Details")
    
    input_data = {}

    # Iterate over schema to create inputs
    for col, values in DATA_SCHEMA.items():
        if isinstance(values, list):
            # Sort options for better UX where appropriate, but respect schema order if meaningful? 
            # Alphabetical is usually safe.
            # But converting numbers to strings for display might be needed ifmixed.
            # Here values are already mostly consistent.
            
            # Sort if all strings, or all numbers
            try:
                options = sorted(values)
            except:
                options = values
                
            input_data[col] = st.sidebar.selectbox(f"{col}", options)
        elif isinstance(values, dict) and values.get('type') == 'number':
            # Numeric Input
            val = float(values['median'])
            min_v = float(values['min'])
            max_v = float(values['max'])
            input_data[col] = st.sidebar.number_input(f"{col}", min_value=min_v, max_value=max_v, value=val)

    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data])
    
    # Ensure columns are in the same/correct order? 
    # The pipeline column transformer usually handles column reordering by name if named, 
    # but strictly speaking, we passed columns by name in `analysis_and_training.py`.
    # `pd.DataFrame([input_data])` will have keys in insertion order (Py3.7+) which matches schema definition order.
    # To be safe, reindex?
    # The schema keys are the exact feature names.
    
    st.subheader("Prediction Result")
    
    if st.button("Predict Fraud Status"):
        try:
            # Predict
            # Pipeline expects specific columns.
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
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check inputs.")

if __name__ == "__main__":
    main()
