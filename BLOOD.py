from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and the scaler
model = pickle.load(open("Model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # <-- You must generate and include this file

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs in the exact order used during training
        months_last = float(request.form['months_last'])
        num_donations = float(request.form['num_donations'])
        total_volume = float(request.form['total_volume'])
        months_first = float(request.form['months_first'])

        # Combine into a numpy array
        final_features = np.array([[months_last, num_donations, total_volume, months_first]])

        # Apply the same scaling used during training
        final_features_scaled = scaler.transform(final_features)

        # Predict using the trained model
        prediction = model.predict(final_features_scaled)[0]

        # Format output for display
        output = "🩸 Will Donate" if prediction == 1 else "🚫 Will Not Donate"

        return render_template("index.html", prediction_text=f"Prediction: {output}")

    except Exception as e:
        # Display error in case of missing or invalid input
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
