from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained AdaBoost model
try:
    model = pickle.load(open("Model.pkl", "rb"))
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template("index.html", prediction_text="Error: Model not loaded")

    try:
        # Collect input features from the form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Predict using the AdaBoost model
        prediction = model.predict(final_features)[0]

        # Determine the prediction result
        output = "Will Donate" if prediction == 1 else "Will Not Donate"

        return render_template("index.html", prediction_text=f"Prediction: {output}")

    except Exception as e:
        # Handle any unexpected error gracefully
        return render_template("index.html", prediction_text=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
