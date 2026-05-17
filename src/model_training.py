"""
Model Training Module
Trains 6 independent Random Forest classifiers (one per nutritional need)
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd

def train_random_forest_model(X_train, y_train, X_test, y_test, **kwargs):
    """Train a single Random Forest classifier"""
    
    rf = RandomForestClassifier(
        n_estimators=kwargs.get('n_estimators', 200),
        max_depth=kwargs.get('max_depth', 10),
        min_samples_split=kwargs.get('min_samples_split', 5),
        random_state=kwargs.get('random_state', 42),
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    pred = rf.predict(X_test)
    prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    
    return rf, pred, prob, acc, auc

def train_all_models(X_train, X_test, Y_train, Y_test, targets, **kwargs):
    """Train 6 independent Random Forest models (one per target)"""
    
    models = {}
    predictions = {}
    probabilities = {}
    results = []
    
    for target in targets:
        print(f"Training model for: {target}")
        
        rf, pred, prob, acc, auc = train_random_forest_model(
            X_train, Y_train[target], X_test, Y_test[target], **kwargs
        )
        
        models[target] = rf
        predictions[target] = pred
        probabilities[target] = prob
        
        results.append({
            'Target': target.replace('_need', '').title(),
            'Accuracy': acc,
            'ROC-AUC': auc
        })
        
        print(f"  ✅ Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
    
    results_df = pd.DataFrame(results)
    avg_acc = results_df['Accuracy'].mean()
    
    print(f"\n📊 Average Accuracy: {avg_acc:.2%}")
    
    return models, predictions, probabilities, results_df, avg_acc
