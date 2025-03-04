from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, static_folder="../frontend/static", template_folder="../frontend")

# Load the trained model and encoders
model = joblib.load("models/sales_model.pkl")
label_encoder_day = joblib.load("models/label_encoder_day.pkl")
label_encoder_dish = joblib.load("models/label_encoder_dish.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    print("Received Data:", data)  # Debugging line

    day = data.get("day")
    dish = data.get("dish")

    try:
        day_encoded = label_encoder_day.transform([day])[0]
        dish_encoded = label_encoder_dish.transform([dish])[0]
    except ValueError:
        return jsonify({"error": "Invalid day or dish"}), 400

    features = np.array([[day_encoded, dish_encoded]])
    predicted_sales = model.predict(features)[0]

    return jsonify({"predicted_sales": round(predicted_sales, 2)})

if __name__ == "__main__":
    app.run(debug=True)
