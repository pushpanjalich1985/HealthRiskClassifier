# ─────────────────────────────────────────────
#  generate_data.py
#  Generates a synthetic health dataset and
#  saves it to data/health_data.csv
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import os


def generate_health_data(n_samples=1000, seed=42):
    """
    Generate a synthetic health dataset.

    Parameters:
        n_samples (int): Number of rows to generate. Default is 1000.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        df (pd.DataFrame): The complete dataset including labels.
    """

    # ── Step 1: Fix the random seed ──────────────────────────────────────────
    # This ensures anyone who runs this code gets identical data
    np.random.seed(seed)


    # ── Step 2: Generate each feature ────────────────────────────────────────

    # Age: whole numbers, uniformly spread between 18 and 90
    age = np.random.randint(18, 91, size=n_samples)

    # BMI: bell curve centred at 27 (slightly overweight average)
    # std=6 gives realistic spread. We clip so no impossible values appear
    bmi = np.random.normal(loc=27, scale=6, size=n_samples)
    bmi = np.clip(bmi, 10, 50)   # clip removes values outside [10, 50]
    bmi = np.round(bmi, 1)       # round to 1 decimal place

    # Blood Pressure (systolic): bell curve around 120 mmHg
    blood_pressure = np.random.normal(loc=120, scale=20, size=n_samples)
    blood_pressure = np.clip(blood_pressure, 80, 200)
    blood_pressure = np.round(blood_pressure).astype(int)  # whole numbers only

    # Cholesterol: bell curve around 200 mg/dL
    cholesterol = np.random.normal(loc=200, scale=40, size=n_samples)
    cholesterol = np.clip(cholesterol, 100, 400)
    cholesterol = np.round(cholesterol).astype(int)

    # Blood Sugar (fasting glucose): bell curve around 100 mg/dL
    blood_sugar = np.random.normal(loc=100, scale=30, size=n_samples)
    blood_sugar = np.clip(blood_sugar, 70, 300)
    blood_sugar = np.round(blood_sugar).astype(int)

    # Smoking: binary 0 or 1. Real-world ~30% of adults smoke
    smoking = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])

    # Physical Activity: hours of exercise per week
    # Skewed — most people do 0-5 hours, fewer do more
    physical_activity = np.random.exponential(scale=3, size=n_samples)
    physical_activity = np.clip(physical_activity, 0, 20)
    physical_activity = np.round(physical_activity, 1)

    # Sleep Hours: bell curve around 7 hours
    sleep_hours = np.random.normal(loc=7, scale=1.5, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 3, 10)
    sleep_hours = np.round(sleep_hours, 1)


    # ── Step 3: Assign risk labels ────────────────────────────────────────────
    # We score each patient based on clinical risk factors.
    # Each dangerous condition adds 1 point to their risk score.
    # Score 0-1 → Low, Score 2-3 → Medium, Score 4+ → High

    # Build a risk score array starting at 0 for everyone
    risk_score = np.zeros(n_samples, dtype=int)

    # Each condition below is a boolean array — True(1) or False(0) per patient
    risk_score += (bmi > 30).astype(int)               # obese
    risk_score += (blood_pressure > 140).astype(int)   # hypertension
    risk_score += (cholesterol > 240).astype(int)      # high cholesterol
    risk_score += (blood_sugar > 126).astype(int)      # diabetic range
    risk_score += (smoking == 1).astype(int)           # smoker
    risk_score += (age > 60).astype(int)               # older age
    risk_score += (physical_activity < 1.5).astype(int) # very inactive

    # Map score to label using np.select — like a multi-condition if/elif/else
    conditions = [
        risk_score <= 1,                        # Low
        (risk_score >= 2) & (risk_score <= 3),  # Medium
        risk_score >= 4                         # High
    ]
    labels = ["Low", "Medium", "High"]
    risk_label = np.select(conditions, labels)


    # ── Step 4: Pack everything into a DataFrame ──────────────────────────────
    df = pd.DataFrame({
        "Age":              age,
        "BMI":              bmi,
        "BloodPressure":    blood_pressure,
        "Cholesterol":      cholesterol,
        "BloodSugar":       blood_sugar,
        "Smoking":          smoking,
        "PhysicalActivity": physical_activity,
        "SleepHours":       sleep_hours,
        "RiskLabel":        risk_label
    })

    return df


def save_data(df, path="data/health_data.csv"):
    """
    Save the DataFrame to a CSV file.
    Creates the folder automatically if it doesn't exist.
    """
    # os.makedirs creates the folder — exist_ok=True means no error if it exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nLabel distribution:\n{df['RiskLabel'].value_counts()}")


# ── This block only runs when you execute THIS file directly ──────────────────
# If another file imports generate_health_data(), this block is skipped
if __name__ == "__main__":
    df = generate_health_data()
    save_data(df)
    print("\nFirst 5 rows:")
    print(df.head())