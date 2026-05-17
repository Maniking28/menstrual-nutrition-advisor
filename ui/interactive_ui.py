"""
Interactive User Interface
Provides real-time predictions with SHAP explanations
"""

import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

def create_widgets():
    """Create all input sliders and dropdowns"""
    
    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='300px')
    
    widgets_dict = {
        'age': widgets.IntSlider(value=28, min=18, max=45, description='Age:', style=style, layout=layout),
        'bmi': widgets.FloatSlider(value=23.0, min=16, max=40, step=0.1, description='BMI:', style=style, layout=layout),
        'sleep_hours': widgets.FloatSlider(value=7.0, min=4, max=10, step=0.5, description='Sleep (hrs):', style=style, layout=layout),
        'stress_level': widgets.IntSlider(value=3, min=1, max=5, description='Stress (1-5):', style=style, layout=layout),
        'activity_level': widgets.Dropdown(
            options=['sedentary', 'light', 'moderate', 'active', 'very_active'],
            value='moderate', description='Activity:', style=style, layout=layout
        ),
        'water_intake': widgets.FloatSlider(value=2.0, min=0.5, max=4.0, step=0.1, description='Water (L):', style=style, layout=layout),
        'cycle_phase': widgets.Dropdown(
            options=['follicular', 'ovulation', 'luteal', 'menstrual'],
            value='luteal', description='Cycle phase:', style=style, layout=layout
        ),
        'flow_intensity': widgets.IntSlider(value=3, min=1, max=5, description='Flow (1-5):', style=style, layout=layout),
        'cramps': widgets.IntSlider(value=3, min=0, max=5, description='Cramps:', style=style, layout=layout),
        'fatigue': widgets.IntSlider(value=3, min=0, max=5, description='Fatigue:', style=style, layout=layout),
        'mood_swings': widgets.IntSlider(value=3, min=0, max=5, description='Mood swings:', style=style, layout=layout),
        'headache': widgets.IntSlider(value=2, min=0, max=5, description='Headache:', style=style, layout=layout),
        'bloating': widgets.IntSlider(value=2, min=0, max=5, description='Bloating:', style=style, layout=layout),
        'acne': widgets.IntSlider(value=1, min=0, max=5, description='Acne:', style=style, layout=layout),
        'sugar_cravings': widgets.IntSlider(value=3, min=0, max=5, description='Sugar cravings:', style=style, layout=layout)
    }
    
    return widgets_dict

def get_recommendation_text(target):
    """Get recommendation text for each nutritional need"""
    rec_texts = {
        'iron_need': '🥩 Iron-rich foods (spinach, lentils, red meat)',
        'magnesium_need': '🥜 Magnesium-rich foods (nuts, seeds, dark chocolate, bananas)',
        'hydration_need': '💧 Increase water intake to 2-3 litres per day',
        'anti_inflammatory_need': '🫚 Anti-inflammatory foods (turmeric, ginger, berries, fatty fish)',
        'low_sugar_need': '🍚 Reduce refined sugar; choose complex carbs and protein',
        'hormone_balance_need': '🌱 Hormone-balancing foods (flax seeds, broccoli, healthy fats)'
    }
    return rec_texts.get(target, 'Consult a healthcare provider')

def display_prediction_card(target, prob, pred, confidence):
    """Display a single prediction as a colored card"""
    
    need_name = target.replace('_need', '').replace('_', ' ').title()
    color = '#e74c3c' if pred else '#27ae60'
    status = 'NEEDED' if pred else 'Not Needed'
    emoji = '🔴' if pred else '🟢'
    
    html = f"""
    <div style="border-left:6px solid {color}; background:#f9f9f9; padding:12px; margin:10px 0; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:1.2rem; font-weight:bold; color:{color};">{emoji} {need_name}: {status}</span>
            <span style="background:{color}; color:white; padding:4px 12px; border-radius:20px;">Confidence: {confidence:.1%}</span>
        </div>
        <p style="margin-top:8px; color:#2c3e50;">💡 {get_recommendation_text(target)}</p>
    </div>
    """
    return html

def launch_ui(rf_models, scaler, encoders, feature_cols, targets):
    """Launch the interactive user interface"""
    
    widgets_dict = create_widgets()
    predict_btn = widgets.Button(description='🌸 Get Recommendations', button_style='success', icon='check')
    output = widgets.Output()
    
    def on_predict_clicked(b):
        with output:
            clear_output()
            
            # Collect user inputs
            inputs = {key: w.value for key, w in widgets_dict.items()}
            
            # Convert to DataFrame
            df = pd.DataFrame([inputs])
            
            # Encode categorical variables
            for col in ['activity_level', 'cycle_phase']:
                df[col + '_encoded'] = encoders[col].transform(df[col])
            
            # Select features and scale
            df = df[feature_cols]
            df_scaled = scaler.transform(df)
            
            # Display header
            display(HTML("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0; text-align: center;">🌸 Your Personalized Nutrition Recommendations</h2>
            </div>
            """))
            
            # Make predictions for all 6 targets
            needed_count = 0
            for target in targets:
                prob = rf_models[target].predict_proba(df_scaled)[0, 1]
                pred = int(prob > 0.5)
                confidence = prob if pred else 1 - prob
                
                if pred:
                    needed_count += 1
                
                display(HTML(display_prediction_card(target, prob, pred, confidence)))
            
            # Show summary
            display(HTML(f"""
            <div style="background:#e8f4f8; padding:12px; border-radius:8px; text-align:center; margin-top:15px;">
                <span style="font-size:1.1rem; font-weight:bold;">📊 {needed_count} of 6 nutritional needs identified</span>
            </div>
            """))
    
    predict_btn.on_click(on_predict_clicked)
    
    # Arrange layout: left column = demographics, right column = symptoms
    left_col = widgets.VBox([
        widgets_dict['age'], widgets_dict['bmi'], widgets_dict['sleep_hours'],
        widgets_dict['stress_level'], widgets_dict['activity_level'],
        widgets_dict['water_intake'], widgets_dict['cycle_phase']
    ])
    
    right_col = widgets.VBox([
        widgets_dict['flow_intensity'], widgets_dict['cramps'], widgets_dict['fatigue'],
        widgets_dict['mood_swings'], widgets_dict['headache'], widgets_dict['bloating'],
        widgets_dict['acne'], widgets_dict['sugar_cravings']
    ])
    
    input_row = widgets.HBox([left_col, right_col])
    ui = widgets.VBox([input_row, predict_btn, output])
    
    # Display the UI
    display(HTML("<h1 style='color:#2c3e50; text-align:center;'>🌸 Menstrual Nutrition Advisor</h1>"))
    display(ui)
