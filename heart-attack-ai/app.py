from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # <- Pozwala na dostęp z Angulara

# Wczytaj model
model = joblib.load('rf_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        X = preprocessor.transform(df)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        return jsonify({'prediction': int(pred), 'probability': round(proba, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)