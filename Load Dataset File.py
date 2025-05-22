# Create the intrusion detection system
ids = IntrusionDetectionSystem()

# Run the full workflow with a single command
results = ids.perform_full_workflow("path/to/your_dataset.csv", "model_save_path.pkl")

# Or use individual methods for more control
ids.load_data("path/to/your_dataset.csv")
ids.identify_columns()
ids.preprocess_data()
ids.train_model()
evaluation = ids.evaluate_model()