"""
Data Generation Module
Creates synthetic dataset using CTGAN (Conditional Tabular GAN)
No real patient data is used - all data is artificially generated
"""

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def create_seed_dataset(n_samples=5000, random_state=42):
    """Create a small seed dataset based on clinical rules"""
    
    np.random.seed(random_state)
    
    data = pd.DataFrame({
        # Demographics
        'age': np.random.randint(18, 45, n_samples),
        'bmi': np.round(np.random.normal(23, 4, n_samples), 1),
        
        # Lifestyle
        'sleep_hours': np.round(np.random.uniform(4, 10, n_samples), 1),
        'stress_level': np.random.randint(1, 6, n_samples),
        'activity_level': np.random.choice(
            ['sedentary', 'light', 'moderate', 'active', 'very_active'], n_samples
        ),
        'water_intake': np.round(np.random.uniform(0.5, 4.0, n_samples), 1),
        
        # Menstrual indicators
        'cycle_phase': np.random.choice(
            ['follicular', 'ovulation', 'luteal', 'menstrual'], n_samples
        ),
        'flow_intensity': np.random.randint(1, 6, n_samples),
        
        # Symptoms (0-5 scale)
        'cramps': np.random.randint(0, 6, n_samples),
        'fatigue': np.random.randint(0, 6, n_samples),
        'mood_swings': np.random.randint(0, 6, n_samples),
        'headache': np.random.randint(0, 6, n_samples),
        'bloating': np.random.randint(0, 6, n_samples),
        'acne': np.random.randint(0, 6, n_samples),
        'sugar_cravings': np.random.randint(0, 6, n_samples)
    })
    
    return data

def add_noise(labels, noise_level=0.15):
    """Add random noise to labels (flip 15% of labels randomly)"""
    mask = np.random.random(len(labels)) < noise_level
    noisy = labels.copy()
    noisy[mask] = 1 - noisy[mask]
    return noisy

def create_labels(data):
    """Create nutritional need labels using clinical rules from medical literature"""
    
    # Iron need: heavy flow OR (fatigue AND heavy flow)
    data['iron_need'] = add_noise(
        ((data['flow_intensity'] >= 4) | 
         ((data['fatigue'] >= 4) & (data['flow_intensity'] >= 3))).astype(int)
    )
    
    # Magnesium need: severe cramps OR (mood swings AND high stress)
    data['magnesium_need'] = add_noise(
        ((data['cramps'] >= 4) | 
         ((data['mood_swings'] >= 4) & (data['stress_level'] >= 4))).astype(int)
    )
    
    # Hydration need: low water intake OR (bloating AND cramps)
    data['hydration_need'] = add_noise(
        ((data['water_intake'] < 2) | 
         ((data['bloating'] >= 3) & (data['cramps'] >= 3))).astype(int)
    )
    
    # Anti-inflammatory need: cramps OR headache OR acne
    data['anti_inflammatory_need'] = add_noise(
        ((data['cramps'] >= 3) | (data['headache'] >= 3) | (data['acne'] >= 3)).astype(int)
    )
    
    # Low sugar need: sugar cravings OR (fatigue AND mood swings)
    data['low_sugar_need'] = add_noise(
        ((data['sugar_cravings'] >= 4) | 
         ((data['fatigue'] >= 3) & (data['mood_swings'] >= 3))).astype(int)
    )
    
    # Hormone balance need: (mood swings during luteal) OR acne
    data['hormone_balance_need'] = add_noise(
        (((data['mood_swings'] >= 3) & (data['cycle_phase'] == 'luteal')) | 
         (data['acne'] >= 3)).astype(int)
    )
    
    # List of target variables
    targets = ['iron_need', 'magnesium_need', 'hydration_need',
               'anti_inflammatory_need', 'low_sugar_need', 'hormone_balance_need']
    
    return data, targets

def generate_synthetic_data(seed_data, n_samples=100000, epochs=300):
    """Generate large synthetic dataset using CTGAN"""
    
    print(f"Training CTGAN on {len(seed_data)} seed samples...")
    print(f"This will take 2-3 minutes...")
    
    # Create metadata (describes data types for CTGAN)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(seed_data)
    
    # Initialize and train CTGAN
    ctgan = CTGANSynthesizer(metadata, epochs=epochs)
    ctgan.fit(seed_data)
    
    # Generate synthetic data
    synthetic_data = ctgan.sample(n_samples)
    
    print(f"✅ Generated {len(synthetic_data)} synthetic samples")
    
    return synthetic_data, ctgan
