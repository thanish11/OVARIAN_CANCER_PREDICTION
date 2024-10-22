from flask_cors import CORS
from flask import Flask, render_template, request, redirect, url_for, session
from src.pipeline.predict_pipeline import PredictionPipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import webbrowser
from threading import Timer
from src.chatbot.models import OvarianCancerChatbot
from langchain_community.document_loaders import PyPDFLoader


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models for premeno and postmeno
premeno_model = PredictionPipeline("models/xgb_model_pre.pkl", "models/scaler_pre.pkl")
postmeno_model = PredictionPipeline("models/xgb_model_post.pkl", "models/scaler_post.pkl")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

@app.route('/')
def home():
    session.clear()
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
                    'CA125': float(request.form.get('CA125', 0)),
                    'HE4': float(request.form.get('HE4', 0)),
                    'CEA': float(request.form.get('CEA', 0)),
                    'ALB': float(request.form.get('ALB', 0)),
                    'TP': float(request.form.get('TP', 0)),
                    'LYM':float(request.form.get('LYM', 0))
                }

                input_data = pd.DataFrame([user_input])

                scaled_data = premeno_model.scaler.transform(input_data)
                prediction = premeno_model.predict(scaled_data)

            elif cancer_type == 'postmenopausal':
                # Collect postmenopausal inputs
                user_input = {
                    'CA125': float(request.form.get('CA125', 0)),
                    'CA199': float(request.form.get('CA199', 0)),
                    'RBC': float(request.form.get('RBC', 0)),
                    'Na':float(request.form.get('Na', 0)),
                    'MONO1':float(request.form.get('MONO1', 0)),
                    'K': float(request.form.get('K', 0)),
                    'HCT': float(request.form.get('HCT', 0))
                }

                input_data = pd.DataFrame([user_input])
                scaled_data = postmeno_model.scaler.transform(input_data)
                prediction = postmeno_model.predict(scaled_data)

            # If prediction is positive (1 or cancer detected), redirect to the chatbot
            # if prediction[0] == 1:
            #     return redirect(url_for('chatbot'))

            return render_template('index.html', prediction=prediction[0])

        except ValueError as ve:
            return f"ValueError: {str(ve)}", 400
        except Exception as e:
            return f"An error occurred: {str(e)}", 500

@app.route('/chat', methods=['GET','POST'])
def chat():
    bot = OvarianCancerChatbot()
    
    
    if 'chat_history' not in session:
        session['chat_history'] = []  # Initialize chat history

    if request.method == 'POST':
        user_input = request.form.get('query', '').strip().lower()  # Extract user query
        session['chat_history'].append(f"You: {user_input}")

        # Store biomarkers in session
        if 'ca125' not in session:
            ca125 = request.form.get('CA125', None)
            if ca125:
                try:
                    session['ca125'] = float(ca125)  # Accept float
                except ValueError:
                    session['chat_history'].append("Bot: Please enter a valid float value for CA125.")
                    return render_template('chat.html', chat_history=session['chat_history'])

        if 'he4' not in session:
            he4 = request.form.get('HE4', None)
            if he4:
                try:
                    session['he4'] = float(he4)  # Accept float
                except ValueError:
                    session['chat_history'].append("Bot: Please enter a valid float value for HE4.")
                    return render_template('chat.html', chat_history=session['chat_history'])

        if 'menopause_status' not in session:
            menopause_status = request.form.get('menopause_status', '').strip().lower()
            if menopause_status not in ["premenopause", "postmenopause"]:
                session['chat_history'].append("Bot: Invalid menopause status. Please provide 'premenopause' or 'postmenopause'.")
                return render_template('chat.html', chat_history=session['chat_history'])
            session['menopause_status'] = menopause_status

        # If biomarkers are all collected, calculate the ROMA score
        if 'ca125' in session and 'he4' in session and 'menopause_status' in session:
            try:
                roma_score = bot.calculate_roma_score(session['ca125'], session['he4'], session['menopause_status'])
                roma_message = f"Your ROMA score is {roma_score:.2f}%."
                if roma_score >= 7.4:
                    roma_message += " High risk of ovarian cancer. Please consult a specialist."
                else:
                    roma_message += " Low risk, but regular monitoring is recommended."
                session['chat_history'].append(f"Bot: {roma_message}")
            except ValueError:
                session['chat_history'].append('Bot: Invalid input for CA125 or HE4.')

        # Respond to general user queries
        response = ""
        if "hospital" in user_input:
            response = bot.query_pdf(bot.hospital_store, user_input)
        elif "diet" in user_input:
            response = bot.query_pdf(bot.diet_store, user_input)
        elif "ovarian" in user_input:
            response = bot.query_pdf(bot.common_store, user_input)
        else:
            # If the query is not related to ovarian cancer
            response = "This is not my domain, but according to my base knowledge, I'll try to assist you."
            response += "\n\n" + bot.answer_general_query(user_input)

        session['chat_history'].append(f"Bot: {response}")
        return render_template('chat.html', chat_history=session['chat_history'])

    return render_template('chat.html', chat_history=session['chat_history'])

# Route to end chat
@app.route('/end_chat')
def end_chat():
    session.clear()  # Clear session to reset the chat
    return redirect(url_for('home'))

# Main function to run the Flask app
if __name__ == '__main__':
    Timer(1, open_browser).start()  
    app.run(debug=True)
