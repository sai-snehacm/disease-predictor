from flask import Flask, render_template, request, redirect, url_for, session, make_response
import joblib
import pandas as pd
from xhtml2pdf import pisa
from io import BytesIO


app = Flask(__name__)
app.secret_key = "mysecretkey"  # Required for sessions

# Dummy user database
users = {"admin": "password123"}

# Load model and helpers
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
symptom_list = joblib.load("symptom_list.pkl")
desc_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")

# ---------------------- Routes ----------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")

        if selected_symptoms:
            input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
            proba = model.predict_proba([input_vector])[0]
            pred_index = proba.argmax()
            prediction = le.inverse_transform([pred_index])[0]
            confidence = round(proba[pred_index] * 100, 2)

            desc_row = desc_df[desc_df["Disease"].str.lower() == prediction.lower()]
            description = desc_row.iloc[0]["Description"] if not desc_row.empty else "No description available."

            precaution_row = precaution_df[precaution_df["Disease"].str.lower() == prediction.lower()]
            precautions = precaution_row.iloc[0][1:].dropna().tolist() if not precaution_row.empty else []

            # Save result to session for report
            session["report"] = {
                "prediction": prediction,
                "confidence": confidence,
                "description": description,
                "precautions": precautions,
                "symptoms": selected_symptoms
            }

            return redirect(url_for("report"))

    return render_template("predict.html", symptoms=symptom_list)

@app.route("/report")
def report():
    if "user" not in session or "report" not in session:
        return redirect(url_for("login"))
    return render_template("report.html", data=session["report"])
@app.route("/download")
def download():
    if "user" not in session or "report" not in session:
        return redirect(url_for("login"))

    # Render HTML from report template
    html = render_template("report.html", data=session["report"], download_mode=True)

    # Convert HTML to PDF
    pdf = BytesIO()
    pisa.CreatePDF(BytesIO(html.encode("utf-8")), dest=pdf)

    # Send PDF as response
    response = make_response(pdf.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
    return response


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------- Run App ----------------------

if __name__ == "__main__":
    app.run(debug=True)
