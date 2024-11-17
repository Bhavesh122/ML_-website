import pickle
from flask import Flask, render_template, request
import pandas as pd

# Load the pickled model and preprocessor
with open('xgb_model.pkl', 'rb') as model_file:
    XGBmodel = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Create Flask app
app = Flask(__name__)

# BMI Calculation function
def calculate_bmi(weight, height):
    return weight / (height / 100) ** 2  # BMI = weight(kg) / height(m)^2

# Prediction function
def predict_cardio_risk(age, height, weight, ap_hi, ap_lo, gender, cholesterol, gluc, smoke, alco, active):
    # Calculate BMI
    bmi = calculate_bmi(weight, height)
    
    # Prepare the input data, including BMI
    input_data = pd.DataFrame({
        'age_years': [age],
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'gender': [gender],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active],
        'bmi': [bmi]  # Include BMI
    })
    
    # Preprocess and scale the input data
    input_data_scaled = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = XGBmodel.predict(input_data_scaled)
    
    return prediction[0]  # 0 or 1

# Route to display the input form
@app.route('/')
def index():
    return render_template('form.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the values from the form
        age = request.form['age']
        height = request.form['height']
        weight = request.form['weight']
        ap_hi = request.form['ap_hi']
        ap_lo = request.form['ap_lo']
        gender = request.form['gender']
        cholesterol = request.form['cholesterol']
        gluc = request.form['gluc']
        smoke = request.form['smoke']
        alco = request.form['alco']
        active = request.form['active']
        
        # Convert values to appropriate types
        age = int(age)
        height = float(height)
        weight = float(weight)
        ap_hi = int(ap_hi)
        ap_lo = int(ap_lo)
        gender = int(gender)
        cholesterol = int(cholesterol)
        gluc = int(gluc)
        smoke = int(smoke)
        alco = int(alco)
        active = int(active)
        
        # Predict the risk
        prediction = predict_cardio_risk(age, height, weight, ap_hi, ap_lo, gender, cholesterol, gluc, smoke, alco, active)
        
        # Customize the result
        if prediction == 1:
            result = "Oops! You have a high chance of cardiovascular disease. Please contact the nearest hospital."
            result_class = "high-risk"
        else:
            result = "Great! You have a low chance of cardiovascular disease. Maintain a healthy lifestyle!"
            result_class = "low-risk"
        
        return render_template('result.html', result=result, result_class=result_class)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}", result_class="error")

if __name__ == '__main__':
    app.run(debug=True)
