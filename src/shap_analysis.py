"""
SHAP Analysis Module
Provides global and local explanations for model predictions
"""

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def compute_global_shap(models, X_sample, feature_cols, targets):
    """Compute global SHAP feature importance for all targets"""
    
    shap_importance = {}
    
    for target in targets:
        print(f"  Computing SHAP for: {target}")
        
        explainer = shap.TreeExplainer(models[target])
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values is a list of two arrays
        if isinstance(shap_values, list):
            shap_values_class = shap_values[1]  # positive class
        else:
            shap_values_class = shap_values
        
        # If still 3D, slice to get class 1
        if shap_values_class.ndim == 3:
            shap_values_class = shap_values_class[..., 1]
        
        # Mean absolute SHAP per feature
        mean_shap = np.abs(shap_values_class).mean(axis=0)
        shap_importance[target] = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)
    
    return shap_importance

def plot_global_shap(shap_importance, targets, save_path=None):
    """Plot global SHAP importance for all targets"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        top_features = shap_importance[target].head(10)
        top_features.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(f'Top features – {target.replace("_need", "").title()}')
        ax.set_xlabel('Mean |SHAP|')
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_waterfall(model, patient_scaled, feature_cols, target_name, save_path=None):
    """Create SHAP waterfall plot for a single patient prediction"""
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_scaled)
    
    # Extract SHAP values for positive class
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    else:
        shap_vals = shap_values[0, :, 1]
        base_val = explainer.expected_value
    
    plt.figure(figsize=(12, 6))
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=patient_scaled[0],
            feature_names=feature_cols
        ),
        show=False,
        max_display=15
    )
    
    plt.title(f'Waterfall Plot - {target_name} Prediction')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
