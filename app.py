from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    prediction_proba = None
    
    if request.method == 'POST':
        # Get form data with default values to prevent NoneType errors
        geography = request.form.get('geography', '')
        gender = request.form.get('gender', '')
        age = int(request.form.get('age', 0))
        balance = float(request.form.get('balance', 0.0))
        credit_score = int(request.form.get('credit_score', 0))
        estimated_salary = float(request.form.get('estimated_salary', 0.0))
        tenure = int(request.form.get('tenure', 0))
        num_of_products = int(request.form.get('num_of_products', 1))
        has_cr_card = int(request.form.get('has_cr_card', 0))
        is_active_member = int(request.form.get('is_active_member', 0))

        # Encode 'Gender'
        try:
            gender_encoded = label_encoder_gender.transform([gender])[0]
        except ValueError:
            gender_encoded = 0  # Default value if unknown

        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [gender_encoded],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode 'Geography'
        if geography in onehot_encoder_geo.categories_[0]:
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        else:
            geo_encoded = np.zeros((1, len(onehot_encoder_geo.categories_[0])))  # Default zero encoding

        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = float(prediction[0][0])
        
        if prediction_proba > 0.5:
            prediction_result = "The customer is likely to churn."
        else:
            prediction_result = "The customer is not likely to churn."
    
    # Get geography and gender options for dropdowns
    geography_options = list(onehot_encoder_geo.categories_[0])
    gender_options = list(label_encoder_gender.classes_)
    
    return render_template('index.html', 
                           prediction_result=prediction_result,
                           prediction_proba=prediction_proba if prediction_proba is not None else 0.0,
                           geography_options=geography_options,
                           gender_options=gender_options)

if __name__ == '__main__':
    app.run(debug=True)
