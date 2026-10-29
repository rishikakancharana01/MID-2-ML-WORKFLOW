"""
Data Generation Script for Synthetic Diabetes Healthcare Dataset.
==================================================================
Purpose:
    Generates a realistic, statistically sound synthetic dataset modeled
    after standard clinical parameters (such as the classic Pima Indians
    Diabetes benchmark) for academic and educational ML assignments.

IMPORTANT DISCLAIMER:
    This dataset is strictly SYNTHETIC and created for educational purposes.
    It does not contain real patient records, protected health information (PHI),
    or personally identifiable information (PII). It must NOT be treated as
    clinical evidence or used for medical diagnoses.
"""

import os
import numpy as np
import pandas as pd


def generate_diabetes_dataset(num_samples: int = 768, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic diabetes records with realistic clinical distributions
    and physiological boundaries.
    """
    np.random.seed(random_state)
    
    # Target distribution: ~65% negative (0), ~35% positive (1)
    n_positive = int(num_samples * 0.349)
    n_negative = num_samples - n_positive
    
    # --- Class 0: Non-Diabetic (Negative) ---
    pregnancies_0 = np.random.negative_binomial(n=2, p=0.45, size=n_negative)
    pregnancies_0 = np.clip(pregnancies_0, 0, 13)
    
    glucose_0 = np.random.normal(loc=110.0, scale=22.0, size=n_negative)
    glucose_0 = np.clip(glucose_0, 70.0, 180.0).round(0)
    
    bp_0 = np.random.normal(loc=70.0, scale=10.0, size=n_negative)
    bp_0 = np.clip(bp_0, 50.0, 106.0).round(0)
    
    skin_0 = np.random.normal(loc=27.0, scale=8.0, size=n_negative)
    skin_0 = np.clip(skin_0, 10.0, 50.0).round(0)
    
    insulin_0 = np.random.normal(loc=125.0, scale=55.0, size=n_negative)
    insulin_0 = np.clip(insulin_0, 20.0, 320.0).round(0)
    
    bmi_0 = np.random.normal(loc=30.3, scale=5.8, size=n_negative)
    bmi_0 = np.clip(bmi_0, 18.5, 48.0).round(1)
    
    dpf_0 = np.random.gamma(shape=2.5, scale=0.17, size=n_negative)
    dpf_0 = np.clip(dpf_0, 0.08, 2.2).round(3)
    
    age_0 = np.random.normal(loc=31.0, scale=10.0, size=n_negative)
    age_0 = np.clip(age_0, 21.0, 75.0).round(0)
    
    outcome_0 = np.zeros(n_negative, dtype=int)

    # --- Class 1: Diabetic (Positive) ---
    pregnancies_1 = np.random.negative_binomial(n=3, p=0.40, size=n_positive)
    pregnancies_1 = np.clip(pregnancies_1, 0, 15)
    
    glucose_1 = np.random.normal(loc=142.0, scale=28.0, size=n_positive)
    glucose_1 = np.clip(glucose_1, 85.0, 200.0).round(0)
    
    bp_1 = np.random.normal(loc=75.0, scale=11.5, size=n_positive)
    bp_1 = np.clip(bp_1, 56.0, 114.0).round(0)
    
    skin_1 = np.random.normal(loc=33.0, scale=8.5, size=n_positive)
    skin_1 = np.clip(skin_1, 12.0, 54.0).round(0)
    
    insulin_1 = np.random.normal(loc=175.0, scale=75.0, size=n_positive)
    insulin_1 = np.clip(insulin_1, 30.0, 420.0).round(0)
    
    bmi_1 = np.random.normal(loc=35.4, scale=6.2, size=n_positive)
    bmi_1 = np.clip(bmi_1, 22.0, 55.0).round(1)
    
    dpf_1 = np.random.gamma(shape=3.2, scale=0.18, size=n_positive)
    dpf_1 = np.clip(dpf_1, 0.12, 2.4).round(3)
    
    age_1 = np.random.normal(loc=37.0, scale=11.0, size=n_positive)
    age_1 = np.clip(age_1, 21.0, 78.0).round(0)
    
    outcome_1 = np.ones(n_positive, dtype=int)

    # Concatenate classes
    data = {
        "Pregnancies": np.concatenate([pregnancies_0, pregnancies_1]).astype(int),
        "Glucose": np.concatenate([glucose_0, glucose_1]).astype(int),
        "BloodPressure": np.concatenate([bp_0, bp_1]).astype(int),
        "SkinThickness": np.concatenate([skin_0, skin_1]).astype(int),
        "Insulin": np.concatenate([insulin_0, insulin_1]).astype(int),
        "BMI": np.concatenate([bmi_0, bmi_1]),
        "DiabetesPedigreeFunction": np.concatenate([dpf_0, dpf_1]),
        "Age": np.concatenate([age_0, age_1]).astype(int),
        "Outcome": np.concatenate([outcome_0, outcome_1]).astype(int),
    }

    df = pd.DataFrame(data)
    
    # Shuffle dataset so outcomes are randomly ordered
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "diabetes.csv")
    
    if os.path.exists(output_path):
        print(f"Dataset already exists at: {output_path}")
    else:
        df = generate_diabetes_dataset()
        df.to_csv(output_path, index=False)
        print(f"Successfully generated {len(df)} synthetic records and saved to: {output_path}")

