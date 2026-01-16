from newspaper import Article
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "https://bolddreamez.com",
            "https://www.bolddreamez.com"
        ]
    }
})


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
from newspaper import Article

@app.route("/predict-url", methods=["POST", "OPTIONS"])
def predict_url():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        article = Article(url)
        article.download()
        article.parse()

        text = article.text
        if len(text) < 100:
            return jsonify({"error": "Article too short"}), 400

        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        confidence = max(model.predict_proba(vector)[0])

        return jsonify({
            "title": article.title,
            "prediction": "Fake" if prediction == 0 else "Real",
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



