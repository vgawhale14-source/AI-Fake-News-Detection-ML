from flask import Flask, render_template, request
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    probability = model.predict_proba(news_vector)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 0:
        result = "🔴 Fake News"
    else:
        result = "🟢 Real News"

    return render_template(
    "index.html",
    prediction=result,
    confidence=confidence,
    news=news
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)