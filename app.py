from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and helper objects
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
symptom_list = joblib.load("symptom_list.pkl")

# Load disease information
desc_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    description = ""
    precautions = []

    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")

        if selected_symptoms:
            # Vector for the model
            input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

            # Make prediction
            pred = model.predict([input_vector])[0]
            prediction = le.inverse_transform([pred])[0]

            # Get description
            desc_row = desc_df[desc_df["Disease"].str.lower() == prediction.lower()]
            if not desc_row.empty:
                description = desc_row.iloc[0]["Description"]
            else:
                description = "No description available."

            # Get precautions
            precaution_row = precaution_df[precaution_df["Disease"].str.lower() == prediction.lower()]
            if not precaution_row.empty:
                # Exclude "Disease" column and filter out null values
                precautions = precaution_row.iloc[0][1:].dropna().tolist()
            else:
                precautions = ["No specific precautions found."]
        else:
            prediction = "Please select at least one symptom."

    return render_template("index.html",
                           symptoms=symptom_list,
                           prediction=prediction,
                           description=description,
                           precautions=precautions)

if __name__ == "__main__":
    app.run(debug=True)
