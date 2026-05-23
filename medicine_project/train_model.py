"""
Training Script for Disease Prediction Model
Trains SVM classifier on symptom data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path

# Set base path
BASE = Path(__file__).resolve().parent

def load_training_data():
    """Load and prepare training dataset"""
    print("Loading training data...")
    
    # Try multiple possible locations
    possible_paths = [
        BASE / "Training.csv",
        BASE / "datasets" / "Training.csv",
        BASE / "Data" / "Training.csv"
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found training data at: {path}")
            df = pd.read_csv(path)
            return df
    
    raise FileNotFoundError("Training.csv not found in any expected location")

def prepare_data(df):
    """Prepare features and labels"""
    print("Preparing data...")
    
    # Separate features and target
    # Assuming last column is 'prognosis' (disease label)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"Features shape: {X.shape}")
    print(f"Unique diseases: {y.nunique()}")
    print(f"Sample diseases: {y.unique()[:5]}")
    
    return X, y

def train_model(X_train, y_train):
    """Train SVM classifier"""
    print("Training SVM model...")
    
    # Initialize SVM with RBF kernel
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return accuracy

def save_model(model, filename="svc.pkl"):
    """Save trained model"""
    save_path = BASE / filename
    
    print(f"\nSaving model to: {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")
    
    # Also save to models directory if it exists
    models_dir = BASE / "models"
    if models_dir.exists():
        alt_path = models_dir / filename
        with open(alt_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model also saved to: {alt_path}")

def main():
    """Main training pipeline"""
    try:
        # Load data
        df = load_training_data()
        print(f"Loaded {len(df)} samples")
        
        # Prepare data
        X, y = prepare_data(df)
        
        # Split data
        print("\nSplitting data (80-20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)
        
        # Save model
        save_model(model)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final Accuracy: {accuracy * 100:.2f}%")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()