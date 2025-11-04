from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("Model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_features)[0]

    # Output text
    output = "Will Donate" if prediction == 1 else "Will Not Donate"

    return render_template("index.html", prediction_text=f"Prediction: {output}")

if __name__ == "__main__":
    app.run(debug=True)
