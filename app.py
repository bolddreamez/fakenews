from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return "Fake News Detector API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    conf = max(model.predict_proba(vec)[0])

    return jsonify({
        "prediction": "Fake" if pred == 0 else "Real",
        "confidence": round(conf * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
