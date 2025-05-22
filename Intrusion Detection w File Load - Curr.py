import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import io
import json
import re


class IntelligentDataLoader:
    """Intelligently loads datasets with unknown formats."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the intelligent data loader.
        
        Args:
            verbose: Whether to print detailed information during loading attempts
        """
        self.verbose = verbose
        self.file_path = None
        self.successful_method = None
        self.data = None
    
    def _print(self, message: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def _load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """Attempt to load as CSV with different delimiters."""
        delimiters = [',', ';', '\t', '|', ' ']
        for delimiter in delimiters:
            try:
                data = pd.read_csv(file_path, delimiter=delimiter)
                self._print(f"Successfully loaded as CSV with delimiter '{delimiter}'")
                return data
            except Exception as e:
                self._print(f"Failed to load as CSV with delimiter '{delimiter}': {str(e)}")
        return None
    
    def _load_excel(self, file_path: str) -> Optional[pd.DataFrame]:
        """Attempt to load as Excel file."""
        try:
            data = pd.read_excel(file_path)
            self._print("Successfully loaded as Excel file")
            return data
        except Exception as e:
            self._print(f"Failed to load as Excel file: {str(e)}")
            return None
    
    def _load_json(self, file_path: str) -> Optional[pd.DataFrame]:
        """Attempt to load as JSON file."""
        try:
            data = pd.read_json(file_path)
            self._print("Successfully loaded as JSON file")
            return data
        except Exception as e:
            self._print(f"Failed to load as JSON file: {str(e)}")
            
            # Try loading as JSON lines
            try:
                data = pd.read_json(file_path, lines=True)
                self._print("Successfully loaded as JSON lines file")
                return data
            except Exception as e2:
                self._print(f"Failed to load as JSON lines file: {str(e2)}")
            
            return None
    
    def _load_parquet(self, file_path: str) -> Optional[pd.DataFrame]:
        """Attempt to load as Parquet file."""
        try:
            data = pd.read_parquet(file_path)
            self._print("Successfully loaded as Parquet file")
            return data
        except Exception as e:
            self._print(f"Failed to load as Parquet file: {str(e)}")
            return None
    
    def _load_plaintext(self, file_path: str) -> Optional[pd.DataFrame]:
        """Try to intelligently parse plain text data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                return None
                
            # Try to detect delimiter based on first few lines
            sample_lines = lines[:min(10, len(lines))]
            potential_delimiters = [',', ';', '\t', '|', ' ']
            delimiter_counts = {}
            
            for delimiter in potential_delimiters:
                counts = [line.count(delimiter) for line in sample_lines]
                consistency = len(set(counts))
                delimiter_counts[delimiter] = (np.mean(counts), consistency)
            
            # Select best delimiter (high count, low consistency value)
            best_delimiter = max(delimiter_counts.items(), 
                                key=lambda x: x[1][0] / (x[1][1] if x[1][1] > 0 else 1))
            
            self._print(f"Detected potential delimiter: '{best_delimiter[0]}'")
            
            # Try parsing with the detected delimiter
            try:
                data = pd.read_csv(io.StringIO(''.join(lines)), delimiter=best_delimiter[0])
                self._print(f"Successfully parsed plaintext with delimiter '{best_delimiter[0]}'")
                return data
            except:
                # If the above fails, try to guess the structure
                parsed_lines = []
                for line in lines:
                    parsed_lines.append(line.strip().split(best_delimiter[0]))
                
                # Find the most common length for rows
                lengths = [len(line) for line in parsed_lines]
                most_common_length = max(set(lengths), key=lengths.count)
                
                # Filter to only rows with this length
                filtered_lines = [line for line in parsed_lines if len(line) == most_common_length]
                
                if filtered_lines:
                    # Use first row as header or generate column names
                    headers = filtered_lines[0]
                    data = pd.DataFrame(filtered_lines[1:], columns=headers)
                    self._print("Parsed plaintext with custom method")
                    return data
                
            return None
        except Exception as e:
            self._print(f"Failed to parse plaintext: {str(e)}")
            return None
    
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Intelligently load a dataset from the given file path.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DataFrame if loading was successful, None otherwise
        """
        self.file_path = file_path
        self._print(f"Attempting to load data from: {file_path}")
        
        if not os.path.exists(file_path):
            self._print(f"Error: File {file_path} does not exist")
            return None
            
        # Determine the file extension
        _, ext = os.path.splitext(file_path.lower())
        ext = ext.strip('.')
        
        # Try loading methods based on extension first
        if ext in ['csv', 'txt', 'dat', 'data']:
            data = self._load_csv(file_path)
            if data is not None:
                self.successful_method = f"CSV (with extension .{ext})"
                self.data = data
                return data
                
        if ext in ['xlsx', 'xls']:
            data = self._load_excel(file_path)
            if data is not None:
                self.successful_method = f"Excel (with extension .{ext})"
                self.data = data
                return data
                
        if ext in ['json']:
            data = self._load_json(file_path)
            if data is not None:
                self.successful_method = "JSON"
                self.data = data
                return data
                
        if ext in ['parquet']:
            data = self._load_parquet(file_path)
            if data is not None:
                self.successful_method = "Parquet"
                self.data = data
                return data
                
        # If extension-based loading failed, try all methods
        self._print("Extension-based loading failed. Trying all methods.")
        
        for method_name, method in [
            ("CSV", self._load_csv),
            ("Excel", self._load_excel),
            ("JSON", self._load_json),
            ("Parquet", self._load_parquet),
            ("Plaintext", self._load_plaintext)
        ]:
            data = method(file_path)
            if data is not None:
                self.successful_method = f"{method_name} (without matching extension)"
                self.data = data
                return data
                
        self._print("All loading methods failed. Could not load the dataset.")
        return None
        
    def get_summary(self) -> Dict:
        """Return a summary of the loaded data."""
        if self.data is None:
            return {"status": "No data loaded"}
            
        summary = {
            "status": "Data loaded successfully",
            "method": self.successful_method,
            "file_path": self.file_path,
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "missing_values": self.data.isnull().sum().to_dict(),
            "sample": self.data.head(5).to_dict() if len(self.data) > 0 else {}
        }
        
        return summary


class IntrusionDetectionSystem:
    """Machine learning-based network intrusion detection system."""
    
    def __init__(self):
        """Initialize the intrusion detection system."""
        self.data_loader = IntelligentDataLoader()
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.numerical_columns = []
        self.categorical_columns = []
        self.target_column = None
        
    def load_data(self, file_path: str) -> bool:
        """
        Load the dataset using the intelligent data loader.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            bool: Whether loading was successful
        """
        self.data = self.data_loader.load_data(file_path)
        return self.data is not None
        
    def identify_columns(self) -> None:
        """Automatically identify numerical, categorical, and target columns."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        # Identify potential target columns (look for labels, attack, intrusion)
        potential_target_columns = []
        for col in self.data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['class', 'label', 'target', 'attack', 'intrusion', 'category']):
                potential_target_columns.append(col)
                
        if not potential_target_columns:
            # If no obvious target column, check for columns with few unique values
            for col in self.data.columns:
                if self.data[col].nunique() < 10 and self.data[col].nunique() > 1:
                    potential_target_columns.append(col)
        
        if potential_target_columns:
            # Choose the column with the fewest unique values as the target
            self.target_column = min(potential_target_columns, key=lambda col: self.data[col].nunique())
        else:
            # Default to the last column if no other indication
            self.target_column = self.data.columns[-1]
            
        print(f"Selected target column: {self.target_column}")
        
        # Identify numerical and categorical columns
        for col in self.data.columns:
            if col == self.target_column:
                continue
                
            if self.data[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(self.data[col]):
                self.numerical_columns.append(col)
            else:
                self.categorical_columns.append(col)
                
        print(f"Found {len(self.numerical_columns)} numerical columns and {len(self.categorical_columns)} categorical columns")
        
    def preprocess_data(self) -> None:
        """Preprocess the data for machine learning."""
        if self.data is None or self.target_column is None:
            print("Data not loaded or columns not identified. Please run load_data() and identify_columns() first.")
            return
            
        # Handle missing values
        for col in self.numerical_columns:
            self.data[col] = self.data[col].fillna(self.data[col].median())
            
        for col in self.categorical_columns:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0] if not self.data[col].mode().empty else "unknown")
            
        # Encode categorical variables
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
            
        # Encode target variable if needed
        if not pd.api.types.is_numeric_dtype(self.data[self.target_column]):
            le = LabelEncoder()
            self.data[self.target_column] = le.fit_transform(self.data[self.target_column].astype(str))
            self.label_encoders[self.target_column] = le
            print(f"Target column classes: {list(le.classes_)}")
            
        # Scale numerical features
        if self.numerical_columns:
            self.scaler = StandardScaler()
            self.data[self.numerical_columns] = self.scaler.fit_transform(self.data[self.numerical_columns])
            
        # Prepare training data
        feature_columns = self.numerical_columns + self.categorical_columns
        self.X = self.data[feature_columns]
        self.y = self.data[self.target_column]
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y if len(np.unique(self.y)) > 1 else None
        )
        
        print(f"Data preprocessed. Training set size: {self.X_train.shape}, Test set size: {self.X_test.shape}")
        
    def train_model(self) -> None:
        """Train the intrusion detection model."""
        if self.X_train is None or self.y_train is None:
            print("Data not preprocessed. Please run preprocess_data() first.")
            return
            
        print("Training Random Forest model for intrusion detection...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed")
        
    def evaluate_model(self) -> Dict:
        """
        Evaluate the trained model.
        
        Returns:
            Dict with evaluation metrics
        """
        if self.model is None or self.X_test is None or self.y_test is None:
            print("Model not trained or data not available. Please train the model first.")
            return {}
            
        print("Evaluating model performance...")
        y_pred = self.model.predict(self.X_test)
        
        acc = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Get feature importances
        importances = self.model.feature_importances_
        features = self.X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
        
        results = {
            "accuracy": acc,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "top_features": importance_df.to_dict(orient='records')
        }
        
        print(f"Model accuracy: {acc:.4f}")
        print("Top 5 important features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
            
        return results
        
    def save_model(self, file_path: str) -> bool:
        """
        Save the trained model to a file.
        
        Args:
            file_path: Path to save the model
            
        Returns:
            bool: Whether saving was successful
        """
        if self.model is None:
            print("No trained model available. Please train the model first.")
            return False
            
        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "numerical_columns": self.numerical_columns,
                "categorical_columns": self.categorical_columns,
                "target_column": self.target_column
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            print(f"Model saved successfully to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
            
    def load_model(self, file_path: str) -> bool:
        """
        Load a trained model from a file.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.label_encoders = model_data["label_encoders"]
            self.numerical_columns = model_data["numerical_columns"]
            self.categorical_columns = model_data["categorical_columns"]
            self.target_column = model_data["target_column"]
            
            print(f"Model loaded successfully from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
            
    def predict(self, data: Union[pd.DataFrame, Dict, List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            data: New data to predict on (DataFrame, dict, or list of dicts)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            print("No trained model available. Please train or load a model first.")
            return None, None
            
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
            
        # Check for required columns
        required_columns = self.numerical_columns + self.categorical_columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None, None
            
        # Preprocess the input data the same way training data was processed
        for col in self.categorical_columns:
            if col in data.columns:
                # Convert to string first
                data[col] = data[col].astype(str)
                
                # Handle new categories not seen during training
                encoder = self.label_encoders[col]
                unique_values = data[col].unique()
                unknown_values = [val for val in unique_values if val not in encoder.classes_]
                
                if unknown_values:
                    # Create a mapping for unknown values
                    known_value = encoder.classes_[0]
                    for val in unknown_values:
                        data.loc[data[col] == val, col] = known_value
                        
                # Now encode
                data[col] = encoder.transform(data[col])
                
        # Scale numerical features
        if self.scaler is not None and self.numerical_columns:
            data[self.numerical_columns] = self.scaler.transform(data[self.numerical_columns])
            
        # Make predictions
        feature_columns = self.numerical_columns + self.categorical_columns
        X_pred = data[feature_columns]
        predictions = self.model.predict(X_pred)
        probabilities = self.model.predict_proba(X_pred)
        
        # Convert predictions back to original labels if a label encoder was used
        if self.target_column in self.label_encoders:
            predictions = self.label_encoders[self.target_column].inverse_transform(predictions)
            
        return predictions, probabilities
        
    def plot_confusion_matrix(self) -> None:
        """Plot the confusion matrix for the model predictions."""
        if self.model is None or self.X_test is None or self.y_test is None:
            print("Model not trained or data not available. Please train the model first.")
            return
            
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def plot_feature_importance(self) -> None:
        """Plot the feature importance of the model."""
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return
            
        importances = self.model.feature_importances_
        features = self.X.columns
        
        indices = np.argsort(importances)[-15:]  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()
        
    def perform_full_workflow(self, file_path: str, save_model_path: str = None) -> Dict:
        """
        Perform the full workflow from data loading to model evaluation.
        
        Args:
            file_path: Path to the dataset file
            save_model_path: Optional path to save the trained model
            
        Returns:
            Dict with results
        """
        results = {}
        
        # Step 1: Load data
        print("\n=== Step 1: Loading data ===")
        if not self.load_data(file_path):
            results["status"] = "failed"
            results["error"] = "Failed to load data"
            return results
            
        results["data_summary"] = self.data_loader.get_summary()
        
        # Step 2: Identify columns
        print("\n=== Step 2: Identifying columns ===")
        self.identify_columns()
        results["column_identification"] = {
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "target_column": self.target_column
        }
        
        # Step 3: Preprocess data
        print("\n=== Step 3: Preprocessing data ===")
        try:
            self.preprocess_data()
            results["preprocessing"] = {
                "status": "success",
                "train_size": self.X_train.shape,
                "test_size": self.X_test.shape
            }
        except Exception as e:
            results["preprocessing"] = {
                "status": "failed",
                "error": str(e)
            }
            results["status"] = "failed"
            return results
            
        # Step 4: Train model
        print("\n=== Step 4: Training model ===")
        try:
            self.train_model()
            results["training"] = {"status": "success"}
        except Exception as e:
            results["training"] = {
                "status": "failed",
                "error": str(e)
            }
            results["status"] = "failed"
            return results
            
        # Step 5: Evaluate model
        print("\n=== Step 5: Evaluating model ===")
        evaluation_results = self.evaluate_model()
        results["evaluation"] = evaluation_results
        
        # Step 6: Save model if requested
        if save_model_path:
            print(f"\n=== Step 6: Saving model to {save_model_path} ===")
            save_success = self.save_model(save_model_path)
            results["model_saving"] = {
                "status": "success" if save_success else "failed",
                "path": save_model_path if save_success else None
            }
            
        results["status"] = "success"
        return results


def main():
    """Main function to demonstrate usage."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python intrusion_detection.py <dataset_path> [model_save_path]")
        return
        
    dataset_path = sys.argv[1]
    model_save_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create and run the intrusion detection system
    ids = IntrusionDetectionSystem()
    results = ids.perform_full_workflow(dataset_path, model_save_path)
    
    # Print final results
    print("\n=== Final Results ===")
    if results["status"] == "success":
        print("Intrusion Detection System completed successfully")
        print(f"Model accuracy: {results['evaluation']['accuracy']:.4f}")
    else:
        print(f"Intrusion Detection System failed: {results.get('error', 'Unknown error')}")
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()