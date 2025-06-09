"""
This file loades anonymized patient data from a CSV file for batch
matching to clinical trials.
"""
import pandas as pd

def load_patient_data(filepath):
    """
    Loads patient data from a CSV file.
    Expects columns: Age, Sex, Condition, Keywords (optional).
    """
    df = pd.read_csv(filepath)
    return df

if __name__ == "__main__":
    patients = load_patient_data("data/patient_data.csv")
    print(patients.head())
