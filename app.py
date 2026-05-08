from flask import Flask, request, jsonify
import pickle
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return "✅ GATE Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        df = pd.DataFrame([data])

        # Encode categorical
        for col, le in encoders.items():
            df[col] = le.transform(df[col])

        # Feature Engineering
        sem_cols = ['sem1','sem2','sem3','sem4','sem5','sem6','sem7','sem8']
        mock_cols = ['mock_test_1','mock_test_2','mock_test_3','mock_test_4','mock_test_5']

        df['avg_sem'] = df[sem_cols].mean(axis=1)
        df['avg_mock'] = df[mock_cols].mean(axis=1)

        # Match feature order
        df = df[features]

        # Prediction
        prediction = model.predict(df)

        return jsonify({
            "predicted_gate_score": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)