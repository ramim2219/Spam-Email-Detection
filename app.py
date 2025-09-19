from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["email_text"]
        message_features = vectorizer.transform([message])
        prediction = model.predict(message_features)[0]

        result = "Ham (Safe Email)" if prediction == 1 else "Spam (Suspicious Email)"
        return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
