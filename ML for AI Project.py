import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

class IntrusionDetectionSystem:
    def __init__(self, model_path='ids_model.pkl'):
        """
        Initialize the Intrusion Detection System
        
        Parameters:
        -----------
        model_path : str
            Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load network traffic dataset
        
        Parameters:
        -----------
        filepath : str
            Path to the dataset CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        print(f"Loading data from {filepath}...")
        try:
            # Attempt to load the data
            data = pd.read_csv(filepath)
            print(f"Successfully loaded {data.shape[0]} records with {data.shape[1]} features")
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def explore_data(self, data):
        """
        Perform exploratory data analysis on the dataset
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to explore
        """
        print("\n=== Dataset Overview ===")
        print(f"Dataset shape: {data.shape}")
        print("\nFirst 5 rows:")
        print(data.head())
        
        print("\nData types:")
        print(data.dtypes)
        
        print("\nSummary statistics:")
        print(data.describe())
        
        print("\nMissing values:")
        print(data.isnull().sum())
        
        # Check target distribution
        if 'label' in data.columns:
            print("\nTarget distribution:")
            print(data['label'].value_counts())
            
            plt.figure(figsize=(10, 6))
            sns.countplot(x='label', data=data)
            plt.title('Distribution of Attack Types')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def preprocess_data(self, data, target_column='label'):
        """
        Preprocess the dataset for training
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to preprocess
        target_column : str
            Name of the target column
            
        Returns:
        --------
        X_train, X_test, y_train, y_test
            Preprocessed data splits
        """
        print("\n=== Preprocessing Data ===")
        
        # Separate features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Save feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Features: {len(self.feature_names)} total ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Create preprocessor
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Fit and transform the training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Handle class imbalance with SMOTE
        if len(np.unique(y_train)) > 1:  # Only apply SMOTE if there are at least 2 classes
            print("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=42)
            X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
            print(f"Training set size after SMOTE: {X_train_processed.shape[0]}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the machine learning model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        """
        print("\n=== Training Model ===")
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print("Training Random Forest Classifier...")
        self.model.fit(X_train, y_train)
        print("Model training completed")
        
        # Save the trained model
        self.save_model()
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target
            
        Returns:
        --------
        float
            Accuracy score
        """
        if self.model is None:
            print("Error: Model not trained yet!")
            return None
        
        print("\n=== Evaluating Model ===")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance()
        
        return accuracy
    
    def _plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        """
        if self.model is None or self.feature_names is None:
            return
            
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create a DataFrame for easier handling
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            all_feature_names = self.preprocessor.get_feature_names_out()
        else:
            # Fallback for older scikit-learn versions
            all_feature_names = self.feature_names
            
        if len(importances) != len(all_feature_names):
            print("Warning: Feature names mismatch, skipping feature importance plot")
            return
            
        feature_importance = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n))
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def save_model(self):
        """Save the trained model and preprocessor"""
        if self.model is None:
            print("Error: No model to save!")
            return
            
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a previously trained model"""
        if not os.path.exists(self.model_path):
            print(f"Error: Model file {self.model_path} not found!")
            return False
            
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_names = model_data['feature_names']
            print(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        data : pd.DataFrame
            New data for prediction
            
        Returns:
        --------
        array-like
            Predictions
        """
        if self.model is None:
            print("Error: Model not trained or loaded!")
            return None
            
        # Preprocess the data
        X_processed = self.preprocessor.transform(data)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        # Create a DataFrame for results
        results = pd.DataFrame({
            'prediction': predictions
        })
        
        # Add probability columns for each class
        for i, class_name in enumerate(self.model.classes_):
            results[f'probability_{class_name}'] = probabilities[:, i]
            
        return results
    
    def detect_intrusions(self, data, threshold=0.7):
        """
        Detect potential intrusions in network traffic
        
        Parameters:
        -----------
        data : pd.DataFrame
            Network traffic data
        threshold : float
            Confidence threshold for alerting
            
        Returns:
        --------
        pd.DataFrame
            Detected intrusions with alert levels
        """
        predictions = self.predict(data)
        if predictions is None:
            return None
            
        # Get the highest probability for each prediction
        max_probs = predictions.iloc[:, 1:].max(axis=1)
        
        # Determine alert level based on probability
        alerts = pd.DataFrame({
            'prediction': predictions['prediction'],
            'confidence': max_probs,
            'alert_level': pd.cut(
                max_probs, 
                bins=[0, 0.5, 0.7, 0.9, 1.0],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        })
        
        # Add original data
        results = pd.concat([data.reset_index(drop=True), alerts], axis=1)
        
        # Filter results above threshold
        intrusions = results[results['confidence'] >= threshold]
        
        return intrusions
    
    def run_workflow(self, data_path, target_column='label', load_existing=False):
        """
        Run the complete IDS workflow
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset
        target_column : str
            Name of the target column
        load_existing : bool
            Whether to load an existing model instead of training
            
        Returns:
        --------
        float
            Model accuracy
        """
        # Load the data
        data = self.load_data(data_path)
        if data is None:
            return None
            
        # Explore the data
        self.explore_data(data)
        
        if load_existing:
            # Load existing model
            if not self.load_model():
                print("Could not load existing model. Training new model...")
                load_existing = False
                
        if not load_existing:
            # Preprocess the data
            X_train, X_test, y_train, y_test = self.preprocess_data(data, target_column)
            
            # Train the model
            self.train_model(X_train, y_train)
            
            # Evaluate the model
            accuracy = self.evaluate_model(X_test, y_test)
            
            return accuracy
        
        return None

# Example usage:
if __name__ == "__main__":
    # Initialize the IDS
    ids = IntrusionDetectionSystem()
    
    # Example with KDD Cup 1999 dataset (a common benchmark for IDS systems)
    # You would need to download this dataset or use your own
    # data_path = "kddcup99_data.csv"  # Replace with your dataset path
    
    # For demonstration, let's use a synthetic dataset
    # This section creates a small synthetic dataset for testing
    print("Creating synthetic dataset for demonstration...")
    
    # Number of samples
    n_samples = 1000
    
    # Create features
    np.random.seed(42)
    data = pd.DataFrame({
        'duration': np.random.randint(0, 100, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'ssh', 'smtp'], n_samples),
        'flag': np.random.choice(['SF', 'REJ', 'S0', 'RSTO'], n_samples),
        'src_bytes': np.random.randint(0, 10000, n_samples),
        'dst_bytes': np.random.randint(0, 10000, n_samples),
        'land': np.random.randint(0, 2, n_samples),
        'wrong_fragment': np.random.randint(0, 10, n_samples),
        'urgent': np.random.randint(0, 5, n_samples),
        'hot': np.random.randint(0, 10, n_samples),
        'num_failed_logins': np.random.randint(0, 5, n_samples),
        'logged_in': np.random.randint(0, 2, n_samples),
        'num_compromised': np.random.randint(0, 20, n_samples),
        'root_shell': np.random.randint(0, 2, n_samples),
        'su_attempted': np.random.randint(0, 2, n_samples),
        'num_file_creations': np.random.randint(0, 10, n_samples),
        'num_shells': np.random.randint(0, 5, n_samples),
        'num_access_files': np.random.randint(0, 10, n_samples),
        'is_host_login': np.random.randint(0, 2, n_samples),
        'is_guest_login': np.random.randint(0, 2, n_samples),
        'count': np.random.randint(0, 500, n_samples),
        'srv_count': np.random.randint(0, 500, n_samples),
        'serror_rate': np.random.random(n_samples),
        'srv_serror_rate': np.random.random(n_samples),
        'rerror_rate': np.random.random(n_samples),
        'srv_rerror_rate': np.random.random(n_samples),
        'same_srv_rate': np.random.random(n_samples),
        'diff_srv_rate': np.random.random(n_samples),
        'srv_diff_host_rate': np.random.random(n_samples),
        'dst_host_count': np.random.randint(0, 255, n_samples),
        'dst_host_srv_count': np.random.randint(0, 255, n_samples),
        'dst_host_same_srv_rate': np.random.random(n_samples),
        'dst_host_diff_srv_rate': np.random.random(n_samples),
        'dst_host_same_src_port_rate': np.random.random(n_samples),
        'dst_host_srv_diff_host_rate': np.random.random(n_samples),
        'dst_host_serror_rate': np.random.random(n_samples),
        'dst_host_srv_serror_rate': np.random.random(n_samples),
        'dst_host_rerror_rate': np.random.random(n_samples),
        'dst_host_srv_rerror_rate': np.random.random(n_samples),
    })
    
    # Create labels: normal or one of four attack types
    attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    # Create imbalanced distribution - common in IDS data
    weights = [0.6, 0.2, 0.1, 0.05, 0.05]  # 60% normal, 40% attacks
    data['label'] = np.random.choice(attack_types, n_samples, p=weights)
    
    # Save synthetic data to a temporary file
    temp_data_path = "synthetic_ids_data.csv"
    data.to_csv(temp_data_path, index=False)
    print(f"Saved synthetic dataset to {temp_data_path}")
    
    # Run the workflow
    accuracy = ids.run_workflow(temp_data_path, target_column='label')
    
    print("\n=== Intrusion Detection Demo ===")
    # Create some new traffic data for prediction
    new_traffic = data.drop('label', axis=1).sample(10).reset_index(drop=True)
    
    # Detect intrusions
    intrusions = ids.detect_intrusions(new_traffic, threshold=0.6)
    
    if intrusions is not None and not intrusions.empty:
        print(f"\nDetected {len(intrusions)} potential intrusions:")
        print(intrusions[['prediction', 'confidence', 'alert_level']].head())
    else:
        print("\nNo intrusions detected in sample traffic")
    
    print("\nDemo completed successfully!")