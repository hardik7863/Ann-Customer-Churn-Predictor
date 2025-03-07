
# Customer Churn Prediction

This project is a web application that predicts customer churn using a trained Artificial Neural Network (ANN) model. The application is built using Flask for the backend and Streamlit for the frontend.

## Project Structure

```
.
├── 1-experiments.ipynb
├── 2-prediction.ipynb
├── app.py
├── Churn_Modelling.csv
├── hyperparametertuningann.ipynb
├── label_encoder_gender.pkl
├── logs/
│   ├── fit/
│   │   ├── 20250106-174528/
│   │   │   ├── train/
│   │   │   │   └── events.out.tfevents.1736165733.hardik-Infinix-INBOOK-X1-SLIM.13407.0.v2
│   │   │   ├── validation/
│   │   │   │   └── events.out.tfevents.1736165735.hardik-Infinix-INBOOK-X1-SLIM.13407.1.v2
├── model.h5
├── onehot_encoder_geo.pkl
├── README.md
├── requirements.txt
├── salaryregression.ipynb
├── scaler.pkl
├── StreamlitApp.py
└── templates/
    └── index.html
```

## Setup and Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the Flask application:
    ```sh
    python app.py
    ```

5. Open your browser and go to `http://127.0.0.1:5000/` to access the application.

## Usage

- The application allows users to input customer details such as geography, gender, age, balance, credit score, estimated salary, tenure, number of products, credit card status, and active member status.
- The model predicts the probability of customer churn and displays the result.

## Model Training

- The model is trained using the Churn_Modelling.csv dataset.
- The training process and hyperparameter tuning are documented in the 1-experiments.ipynb and hyperparametertuningann.ipynb notebooks.
- The trained model and encoders are saved as model.h5, label_encoder_gender.pkl, onehot_encoder_geo.pkl, and scaler.pkl.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The project uses TensorFlow and Keras for building and training the ANN model.
- Flask is used for building the web application backend.
- Streamlit is used for building the frontend interface.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
# Ann-Customer-Churn-Predictor
