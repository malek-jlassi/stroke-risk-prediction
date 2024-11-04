import pickle
import pandas as pd

# Load the saved model
with open('stroke_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Optional: Print the expected feature names from the model
try:
    # If you are using a pipeline, extract the feature names from the preprocessor
    feature_names = model.named_steps['classifier'].feature_importances_  # This is to check model readiness, replace it if necessary
    print("Expected feature names:", feature_names)
except Exception as e:
    print("Could not retrieve feature names from the model:", str(e))

# Define test cases with the correct features
test_cases = pd.DataFrame({
    'age': [70, 30],                                      # Sample ages
    'hypertension': [0, 0],                               # Both without hypertension
    'heart_disease': [0, 0],                              # Both without heart disease
    'avg_glucose_level': [80.0, 90.0],                    # Different glucose levels
    'bmi': [35.0, 22.0],                                  # Different BMIs
    'gender_Male': [1, 0],                                # Male and female
    'ever_married_Yes': [1, 0],                           # One married, one unmarried
    'Residence_type_Urban': [1, 0],                       # One in urban, one in rural
    'smoking_status_never smoked': [0, 0],                # Both never smoked
    'smoking_status_smokes': [1, 0],                      # One smokes, one doesn't
    'smoking_status_formerly smoked': [0, 0],             # Both do not smoke formerly
})

# Ensure test_cases matches the feature names used during training
print("Test Cases DataFrame Columns:")
print(test_cases.columns)

# Run predictions
predictions = model.predict(test_cases)

# Display the results
for i, pred in enumerate(predictions):
    result = "Stroke" if pred == 1 else "No Stroke"
    print(f"Test Case {i + 1}: Prediction - {result}")
