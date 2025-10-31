from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model (tuned RandomForest)
model = pickle.load(open("Model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(final_features)[0]

        # Decide output text
        output = "Will Donate" if prediction == 1 else "Will Not Donate"

        return render_template("index.html", prediction_text=f"Prediction: {output}")

    except Exception as e:
        # Handle any unexpected errors
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
