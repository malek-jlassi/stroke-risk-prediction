import pickle
from flask import Flask, render_template, request, redirect, url_for

# Load the trained model
with open('stroke_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    gender_male = 1 if request.form['gender'] == 'Male' else 0
    ever_married_yes = 1 if request.form['ever_married'] == 'Yes' else 0
    residence_type_urban = 1 if request.form['residence_type'] == 'Urban' else 0
    smoking_status = request.form['smoking_status']
    
    smoking_status_never_smoked = 1 if smoking_status == 'never smoked' else 0
    smoking_status_smokes = 1 if smoking_status == 'smokes' else 0
    smoking_status_formerly_smoked = 1 if smoking_status == 'formerly smoked' else 0

    # Prepare the input for the model
    input_features = [[
        age, hypertension, heart_disease, avg_glucose_level, bmi,
        gender_male, ever_married_yes, residence_type_urban,
        smoking_status_never_smoked, smoking_status_smokes,
        smoking_status_formerly_smoked
    ]]

    # Make prediction
    prediction = model.predict(input_features)
    result = "Stroke" if prediction[0] == 1 else "No Stroke"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
