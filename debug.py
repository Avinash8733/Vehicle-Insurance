import sklearn
import joblib
import pandas as pd
import numpy as np

print(f"Scikit-learn version: {sklearn.__version__}")

try:
    pipeline = joblib.load('pipeline.pkl')
    print("Pipeline loaded successfully.")
    
    # Create dummy data based on schema structure
    # We need a dataframe with the right columns
    cols = ['Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed', 'Sex', 'MaritalStatus', 'Age', 'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice', 'RepNumber', 'Deductible', 'DriverRating', 'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars', 'Year', 'BasePolicy']
    
    # Create a dummy row with appropriate values
    data = {
        'Month': 'Jan', 'WeekOfMonth': 1, 'DayOfWeek': 'Monday', 'Make': 'Honda',
        'AccidentArea': 'Urban', 'DayOfWeekClaimed': 'Tuesday', 'MonthClaimed': 'Jan',
        'WeekOfMonthClaimed': 1, 'Sex': 'Male', 'MaritalStatus': 'Single', 'Age': 30,
        'Fault': 'Policy Holder', 'PolicyType': 'Sedan - All Perils', 'VehicleCategory': 'Sedan',
        'VehiclePrice': '20000 to 29000', 'RepNumber': 1, 'Deductible': 400, 'DriverRating': 1,
        'Days_Policy_Accident': 'more than 30', 'Days_Policy_Claim': 'more than 30',
        'PastNumberOfClaims': 'none', 'AgeOfVehicle': '3 years', 'AgeOfPolicyHolder': '31 to 35',
        'PoliceReportFiled': 'No', 'WitnessPresent': 'No', 'AgentType': 'External',
        'NumberOfSuppliments': 'none', 'AddressChange_Claim': 'no change', 'NumberOfCars': '1 vehicle',
        'Year': 1994, 'BasePolicy': 'All Perils'
    }
    
    df = pd.DataFrame([data])
    print("Dummy DataFrame created.")
    
    pred = pipeline.predict(df)
    print(f"Prediction successful: {pred}")

except Exception as e:
    print(f"Error: {e}")
