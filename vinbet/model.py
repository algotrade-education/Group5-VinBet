import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt

def train_lgbm_model(X, y):
    """Trains a LightGBM Classifier."""
    
    # Time-series split (no shuffling!)
    # Train on first 80%, validate on next 20%
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=20)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    return model, X_val, y_val

def evaluate_model(model, X_val, y_val):
    """Evaluates the trained model."""
    # Predict probabilities
    y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
    # Convert to binary class
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_val, y_pred_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return y_pred, y_pred_prob

def plot_importance(model, importance_type='split'):
    """Plots feature importance."""
    lgb.plot_importance(model, importance_type=importance_type, max_num_features=20, figsize=(10, 6))
    plt.title(f"Feature Importance ({importance_type})")
    plt.tight_layout()
    # plt.show() # In a CLI context this might not show, but saves the code.
    # Alternatively save to file
    plt.savefig(f"feature_importance_{importance_type}.png")
    print(f"Feature importance plot saved to feature_importance_{importance_type}.png")
