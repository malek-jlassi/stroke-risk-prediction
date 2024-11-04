import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pickle

# Load your dataset
data = pd.read_csv('healthcare_dataset.csv')

# One-Hot Encoding for categorical features
data = pd.get_dummies(data, columns=['gender', 'ever_married', 'Residence_type', 'smoking_status'], drop_first=False)

# Define features and label
features = [
    'age',
    'hypertension',
    'heart_disease',
    'avg_glucose_level',
    'bmi',
    'gender_Male',                # One-hot encoding result
    'ever_married_Yes',          # One-hot encoding result
    'Residence_type_Urban',       # One-hot encoding result
    'smoking_status_never smoked', # One-hot encoding result
    'smoking_status_smokes',      # One-hot encoding result
    'smoking_status_formerly smoked' # One-hot encoding result
]

# Define Features and Labels
X = data[features]  # Select only the defined features
y = data['stroke']  # Labels (1 for stroke, 0 for no stroke)


# Save the modified features to a new CSV file
X.to_csv('processed_features.csv', index=False)

# Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Function to Create a Pipeline with SMOTE and a Classifier
def create_pipeline(classifier):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
        ('classifier', classifier)
    ])

# Define Parameter Grids for Random Forest
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
}

# Train and Evaluate Random Forest Classifier
pipeline_rf = create_pipeline(RandomForestClassifier(random_state=42))
best_rf_model = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
best_rf_model.fit(X_train, y_train)

# Output the classification report
y_pred_rf = best_rf_model.predict(X_test)
print("Classification Report on Test Set (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Save the model
with open('stroke_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)

print("Model saved successfully.")
