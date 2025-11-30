import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

class ResumeClassifier:
    def __init__(self):
        """Initialize resume classifier with multiple models"""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_logistic_regression(self, X_train, y_train, max_iter=1000):
        """
        Train Logistic Regression classifier
        
        Args:
            X_train: Training features (TF-IDF matrix)
            y_train: Training labels
            max_iter: Maximum iterations
            
        Returns:
            Trained model
        """
        print("\n=== Training Logistic Regression ===")
        
        lr_model = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            solver='liblinear',
            class_weight='balanced'
        )
        
        lr_model.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        
        print("Logistic Regression trained successfully")
        return lr_model
    
    def train_naive_bayes(self, X_train, y_train):
        """
        Train Multinomial Naive Bayes classifier
        
        Args:
            X_train: Training features (TF-IDF matrix)
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print("\n=== Training Naive Bayes ===")
        
        nb_model = MultinomialNB(alpha=1.0)
        nb_model.fit(X_train, y_train)
        self.models['Naive Bayes'] = nb_model
        
        print("Naive Bayes trained successfully")
        return nb_model
    
    def train_all_models(self, X_train, y_train):
        """
        Train all classification models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        self.train_logistic_regression(X_train, y_train)
        self.train_naive_bayes(X_train, y_train)
        
        return self.models
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Print metrics
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'confusion_matrix': cm
        }
        
        return results
    
    def cross_validate(self, model, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"\nCross-Validation Scores: {scores}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def predict(self, model, X):
        """
        Make predictions
        
        Args:
            model: Trained model
            X: Features to predict
            
        Returns:
            Predictions
        """
        return model.predict(X)
    
    def predict_proba(self, model, X):
        """
        Get prediction probabilities
        
        Args:
            model: Trained model
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def classify_resume(self, model, resume_vector, class_names=None):
        """
        Classify a single resume
        
        Args:
            model: Trained model
            resume_vector: TF-IDF vector of resume
            class_names: List of class names (optional)
            
        Returns:
            Predicted class and probability
        """
        prediction = model.predict(resume_vector)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(resume_vector)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None
        
        if class_names:
            predicted_class = class_names[prediction]
        else:
            predicted_class = prediction
        
        return {
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2) if confidence else None,
            'raw_prediction': prediction
        }
    
    def save_model(self, model, filename):
        """Save trained model"""
        joblib.dump(model, filename)
        print(f"\nModel saved to {filename}")
    
    def load_model(self, filename):
        """Load saved model"""
        model = joblib.load(filename)
        print(f"\nModel loaded from {filename}")
        return model
    
    def get_feature_importance(self, model, feature_names, top_n=20):
        """
        Get feature importance for Logistic Regression
        
        Args:
            model: Trained Logistic Regression model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top features per class
        """
        if not isinstance(model, LogisticRegression):
            print("Feature importance only available for Logistic Regression")
            return None
        
        # Get coefficients
        coef = model.coef_
        
        feature_importance = {}
        
        # For each class
        for i, class_coef in enumerate(coef):
            # Get top positive features
            top_positive_idx = np.argsort(class_coef)[-top_n:][::-1]
            top_positive = [(feature_names[idx], class_coef[idx]) for idx in top_positive_idx]
            
            # Get top negative features
            top_negative_idx = np.argsort(class_coef)[:top_n]
            top_negative = [(feature_names[idx], class_coef[idx]) for idx in top_negative_idx]
            
            feature_importance[f'class_{i}'] = {
                'top_positive': top_positive,
                'top_negative': top_negative
            }
        
        return feature_importance