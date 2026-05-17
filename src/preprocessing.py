"""
Preprocessing Module
Handles encoding, feature engineering, scaling, and train-test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def encode_categorical(data, cat_cols=['activity_level', 'cycle_phase']):
    """Encode categorical variables to numbers"""
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col])
        encoders[col] = le
    
    return data, encoders

def engineer_features(data):
    """Create new features from existing ones"""
    
    # Total symptom score (sum of all 7 symptoms)
    data['total_symptom_score'] = (
        data['cramps'] + data['fatigue'] + data['mood_swings'] +
        data['headache'] + data['bloating'] + data['acne'] +
        data['sugar_cravings']
    )
    
    # Pain score (cramps + headache)
    data['pain_score'] = data['cramps'] + data['headache']
    
    # Mood score (mood swings + stress)
    data['mood_score'] = data['mood_swings'] + data['stress_level']
    
    # Interaction term (cramps × flow intensity)
    data['cramps_x_flow'] = data['cramps'] * data['flow_intensity']
    
    return data

def get_features():
    """Return list of all features (original + engineered)"""
    return [
        'age', 'bmi', 'sleep_hours', 'stress_level',
        'activity_level_encoded', 'water_intake', 'cycle_phase_encoded',
        'flow_intensity', 'cramps', 'fatigue', 'mood_swings',
        'headache', 'bloating', 'acne', 'sugar_cravings',
        'total_symptom_score', 'pain_score', 'mood_score', 'cramps_x_flow'
    ]

def preprocess_data(data, targets, test_size=0.2, random_state=42):
    """Complete preprocessing pipeline"""
    
    # Step 1: Encode categorical variables
    data, encoders = encode_categorical(data)
    
    # Step 2: Create engineered features
    data = engineer_features(data)
    
    # Step 3: Get features and targets
    features = get_features()
    X = data[features]
    Y = data[targets]
    
    # Step 4: Scale features (mean=0, variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Step 5: Split into train (80%) and test (20%)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=test_size, 
        random_state=random_state, stratify=Y[targets[0]]
    )
    
    return X_train, X_test, Y_train, Y_test, scaler, encoders, features
