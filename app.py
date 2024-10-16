from flask import Flask, render_template, request, redirect, url_for
from src.pipeline.predict_pipeline import PredictionPipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

# Load models for premeno and postmeno
premeno_model = PredictionPipeline("models/xgboost_premenopausal_model.pkl")
postmeno_model = PredictionPipeline("models/xgboost_postmenopausal_model.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        cancer_type = request.form.get('cancer_type', None)

        if cancer_type is None:
            return "Error: Cancer type not specified", 400
        
        try:
            if cancer_type == 'premenopausal':
                # Collect premenopausal inputs
                user_input = {
                    'AGE': float(request.form.get('Age', 0)),
                    'CA125': float(request.form.get('CA125', 0)),
                    'HE4': float(request.form.get('HE4', 0)),
                    'CEA': float(request.form.get('CEA', 0)),
                    'CA724': float(request.form.get('CA724', 0)),
                    'GLO': float(request.form.get('GLO', 0)),
                    'ALB': float(request.form.get('ALB', 0)),
                    'CA199': float(request.form.get('CA199', 0)),
                    'MCH': float(request.form.get('MCH', 0)),
                    'AFP': float(request.form.get('AFP', 0)),
                    'TP': float(request.form.get('TP', 0)),
                    'AG': float(request.form.get('AG', 0)),
                    'Ca': float(request.form.get('Ca', 0)),
                    'Na': float(request.form.get('Na', 0)),
                    'MPV': float(request.form.get('MPV', 0)),
                    'GLU': float(request.form.get('GLU', 0)),
                    'CL': float(request.form.get('CL', 0)),
                    'NEU': float(request.form.get('NEU', 0)),
                    'UA': float(request.form.get('UA', 0)),
                    'MONO1': float(request.form.get('MONO1', 0)),
                    'ALT': float(request.form.get('ALT', 0))
                }

                input_data = pd.DataFrame([user_input])
                # Apply StandardScaler to the input data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(input_data)
                prediction = premeno_model.predict(scaled_data)

            elif cancer_type == 'postmenopausal':
                # Collect postmenopausal inputs
                user_input = {
                    'AGE': float(request.form.get('Age', 0)),
                    'HE4': float(request.form.get('HE4', 0)),
                    'CA125': float(request.form.get('CA125', 0)),
                    'CA724': float(request.form.get('CA724', 0)),
                    'ALT': float(request.form.get('ALT', 0)),
                    'BUN': float(request.form.get('BUN', 0)),
                    'TBIL': float(request.form.get('TBIL', 0)),
                    'PLT': float(request.form.get('PLT', 0)),
                    'BASO1': float(request.form.get('BASO1', 0)),
                    'PCT': float(request.form.get('PCT', 0)),
                    'Ca': float(request.form.get('Ca', 0)),
                    'K': float(request.form.get('K', 0)),
                    'ALB': float(request.form.get('ALB', 0)),
                    'CEA': float(request.form.get('CEA', 0)),
                    'CO2CP': float(request.form.get('CO2CP', 0)),
                    'CL': float(request.form.get('CL', 0)),
                    'CREA': float(request.form.get('CREA', 0)),
                    'DBIL': float(request.form.get('DBIL', 0)),
                    'UA': float(request.form.get('UA', 0)),
                    'CA199': float(request.form.get('CA199', 0)),
                    'EO1': float(request.form.get('EO1', 0))
                }

                input_data = pd.DataFrame([user_input])
                 # Apply StandardScaler to the input data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(input_data)
                prediction = postmeno_model.predict(scaled_data)

            # If prediction is positive (1 or cancer detected), redirect to the chatbot
            # if prediction[0] == 1:
            #     return redirect(url_for('chatbot'))

            return render_template('index.html', prediction=prediction[0])

        except ValueError as ve:
            return f"ValueError: {str(ve)}", 400
        except Exception as e:
            return f"An error occurred: {str(e)}", 500

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

if __name__ == "__main__":
    app.run(debug=True)
