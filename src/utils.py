"""
Utility Functions
Helper functions for the project
"""

def get_target_names():
    """Return list of target names"""
    return [
        'iron_need', 'magnesium_need', 'hydration_need',
        'anti_inflammatory_need', 'low_sugar_need', 'hormone_balance_need'
    ]

def get_recommendation_text(target):
    """Return recommendation text for each nutritional need"""
    
    recommendations = {
        'iron_need': 'Iron-rich foods (spinach, lentils, red meat)',
        'magnesium_need': 'Magnesium-rich foods (nuts, seeds, dark chocolate, bananas)',
        'hydration_need': 'Increase water intake to 2-3 litres per day',
        'anti_inflammatory_need': 'Anti-inflammatory foods (turmeric, ginger, berries, fatty fish)',
        'low_sugar_need': 'Reduce refined sugar; choose complex carbs and protein',
        'hormone_balance_need': 'Hormone-balancing foods (flax seeds, broccoli, healthy fats)'
    }
    
    return recommendations.get(target, 'Consult a healthcare provider')

def get_sample_patient_severe():
    """Return sample patient with severe symptoms"""
    return {
        'age': 28, 'bmi': 22.5, 'sleep_hours': 5, 'stress_level': 5,
        'activity_level': 'sedentary', 'water_intake': 1.0,
        'cycle_phase': 'luteal', 'flow_intensity': 5,
        'cramps': 5, 'fatigue': 5, 'mood_swings': 5,
        'headache': 4, 'bloating': 5, 'acne': 3, 'sugar_cravings': 5
    }

def get_sample_patient_mild():
    """Return sample patient with mild symptoms"""
    return {
        'age': 32, 'bmi': 23.0, 'sleep_hours': 8, 'stress_level': 2,
        'activity_level': 'active', 'water_intake': 3.0,
        'cycle_phase': 'follicular', 'flow_intensity': 2,
        'cramps': 1, 'fatigue': 2, 'mood_swings': 1,
        'headache': 0, 'bloating': 1, 'acne': 0, 'sugar_cravings': 1
    }
