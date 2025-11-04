from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained AdaBoost model
try:
    model = pickle.load(open("Model.pkl", "rb"))
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template("index.html", prediction_text="Error: Model not loaded properly.")

    try:
        # Collect input features from the form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Get probability of class '1' (Will Donate)
        probabilities = model.predict_proba(final_features)[0]
        donate_prob = probabilities[1]  # probability of class 1

        # Apply threshold
        if donate_prob >= 0.5:
            output = f"Will Donate (Confidence: {donate_prob:.2f})"
        else:
            output = f"Will Not Donate (Confidence: {donate_prob:.2f})"

        return render_template("index.html", prediction_text=f"Prediction: {output}")

    except Exception as e:
        # Handle any unexpected error gracefully
        return render_template("index.html", prediction_text=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
