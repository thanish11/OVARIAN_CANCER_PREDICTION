from flask import Flask, render_template, request, redirect, url_for
from src.components.predict_pipeline import PredictionPipeline
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
        cancer_type = request.form['cancer_type']

        if cancer_type == 'premenopausal':
            # Collect premenopausal inputs
            user_input = {
                'AGE': request.form['age'],
                'CA125': request.form['ca125'],
                'HE4': request.form['he4'],
                'CEA': request.form['cea'],
                'CA724':request.form['CA724']
            }
#['Age', 'HE4', 'CA724', 'CEA', 'GLO', 'ALB', 'CA199', 'MCH', 'AFP', 'TP', 'AG', 'CA125', 'Ca', 'Na', 'MPV', 'GLU', 'CL', 'NEU', 'UA', 'MONO1', 'ALT']
            input_data = pd.DataFrame([user_input])
            prediction = premeno_model.predict(input_data)

        elif cancer_type == 'postmenopausal':
            # Collect postmenopausal inputs
            user_input = {
                'AGE': request.form['age'],
                'CA125': request.form['ca125'],
                'ALB': request.form['alb'],
                'GGT': request.form['ggt'],
                # Add other necessary features for postmenopausal prediction
            }
#['Age', 'HE4', 'CA125', 'CA724', 'ALT', 'BUN', 'TBIL', 'PLT', 'BASO1', 'PCT', 'Ca', 'K', 'ALB', 'CEA', 'CO2CP', 'CL', 'CREA', 'DBIL', 'UA', 'CA199', 'EO1']
            input_data = pd.DataFrame([user_input])
            prediction = postmeno_model.predict(input_data)

        # If prediction is positive (1 or cancer detected), redirect to the chatbot
        if prediction[0] == 1:
            return redirect(url_for('chatbot'))

        return render_template('index.html', prediction=prediction[0])

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

if __name__ == "__main__":
    app.run(debug=True)
