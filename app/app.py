from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load trained model
model_path = "models/spam_classifier.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Run spam_classifier.py first.")

with open(model_path, "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        message = request.form["message"]
        transformed_message = vectorizer.transform([message])
        prediction = model.predict(transformed_message)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
