from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Create the uploads directory if it doesn't exist
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the saved model
model = joblib.load("best_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if a file is uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]

        # Check if the file is empty
        if file.filename == "":
            return jsonify({"error": "No file selected for uploading"}), 400

        # Save the file to the upload directory
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Read the CSV file into a DataFrame
        new_data = pd.read_csv(filepath)

        # Example preprocessing (similar to training script)
        new_data["user_growth_rate"] = (
            new_data["new_users"] - new_data["users_left"]
        ) / new_data["existing_users_before"]

        # Define features
        X_new = new_data[
            [
                "new_users",
                "users_left",
                "existing_users_before",
                "existing_users_after",
                "user_growth_rate",
            ]
        ]

        # Make predictions
        predictions = model.predict(X_new)

        # Map predictions back to labels
        label_mapping = {1: "positive", 0: "negative", 2: "average"}
        predicted_labels = [label_mapping[pred] for pred in predictions]

        # Add predictions to the DataFrame
        new_data["predicted_strategy_effectiveness"] = predicted_labels

        # Aggregate results to give a single output
        positive_count = (
            new_data["predicted_strategy_effectiveness"] == "positive"
        ).sum()
        negative_count = (
            new_data["predicted_strategy_effectiveness"] == "negative"
        ).sum()
        average_count = (
            new_data["predicted_strategy_effectiveness"] == "average"
        ).sum()

        # Determine the overall strategy effectiveness
        if positive_count > negative_count and positive_count > average_count:
            overall_outcome = "Overall Strategy is positive and can be used for the next upcoming years."
        elif negative_count > positive_count and negative_count > average_count:
            overall_outcome = "Overall Strategy isn't working."
        else:
            overall_outcome = "Overall Strategy is working but upgradation is required."

        response = {
            "individual_predictions": new_data.to_dict(orient="records"),
            "overall_outcome": overall_outcome,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
