from pathlib import Path
from flask import Flask, jsonify, request, render_template
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "model.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "model" / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {BASE_DIR / 'Model' / 'model.pkl'} or {BASE_DIR / 'model' / 'model.pkl'}")

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="statics",
    static_url_path="/static",
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FIELD_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "ChestPainType",
    "trestbps": "RestingBP",
    "chol": "Cholesterol",
    "fbs": "FastingBS",
    "restecg": "RestingECG",
    "thalach": "MaxHR",
    "exang": "ExerciseAngina",
    "oldpeak": "Oldpeak",
    "slope": "ST_Slope",
    "ca": "MajorVessels",
    "thal": "Thal",
}


def load_dataset():
    data_dir = BASE_DIR / "Data"
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        return None
    known_names = {"heart.csv", "heart.csv.csv"}
    data_path = next((p for p in csv_files if p.name in known_names), csv_files[0])
    return pd.read_csv(data_path)


DATA = load_dataset()


def build_dashboard_context(form_values=None, selected_row=None, prediction=None):
    form_values = form_values or {}
    context = {
        "form_values": form_values,
        "selected_row": selected_row,
        "prediction": prediction,
        "stats": {
            "split": "70/30",
            "records": 0,
            "train_records": 0,
            "test_records": 0,
            "training_accuracy": None,
            "testing_accuracy": None,
        },
        "sample_rows": [],
        "feature_importance": [],
    }

    if DATA is None or "target" not in DATA.columns:
        return context

    X = DATA[FIELD_NAMES]
    y = DATA["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    context["stats"].update(
        {
            "records": len(DATA),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "training_accuracy": accuracy_score(y_train, model.predict(X_train)) * 100,
            "testing_accuracy": accuracy_score(y_test, model.predict(X_test)) * 100,
        }
    )

    rows = DATA.head(80).reset_index()
    context["sample_rows"] = [
        {
            "index": int(row["index"]),
            "label": f"Row {int(row['index'])} - {'Positive' if row['target'] == 1 else 'Negative'}",
            "values": {field: float(row[field]) for field in FIELD_NAMES},
        }
        for _, row in rows.iterrows()
    ]

    if hasattr(model, "feature_importances_"):
        importances = [
            {"label": FEATURE_LABELS.get(field, field), "value": float(value)}
            for field, value in zip(FIELD_NAMES, model.feature_importances_)
        ]
        max_value = max((item["value"] for item in importances), default=1)
        context["feature_importance"] = sorted(
            [
                {
                    **item,
                    "percent": (item["value"] / max_value) * 100 if max_value else 0,
                }
                for item in importances
            ],
            key=lambda item: item["value"],
        )

    return context


@app.route("/")
def home():
    return render_template("heart_dashboard.html", **build_dashboard_context())


@app.route("/dashboard")
def dashboard():
    return render_template("heart_dashboard.html", **build_dashboard_context())

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        try:
            data = [float(payload[field]) for field in FIELD_NAMES]
        except KeyError as exc:
            return jsonify({"error": f"Missing field: {exc.args[0]}"}), 400
        except (TypeError, ValueError):
            return jsonify({"error": "Please send valid numeric values for all fields."}), 400

        final_data = pd.DataFrame([data], columns=FIELD_NAMES)
        prediction = model.predict(final_data)
        predicted_class = int(prediction[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(final_data)[0][predicted_class] * 100)

        context = build_dashboard_context()
        return jsonify(
            {
                "prediction": "Positive" if predicted_class == 1 else "Negative",
                "probability": probability,
                "feature_importance": {
                    item["label"]: item["value"] for item in context["feature_importance"]
                },
            }
        )

    try:
        data = [float(request.form[field]) for field in FIELD_NAMES]
    except KeyError as exc:
        missing = exc.args[0]
        prediction = {"text": f"Missing field: {missing}", "status": "error"}
        return render_template("heart_dashboard.html", **build_dashboard_context(request.form, prediction=prediction))
    except ValueError:
        prediction = {"text": "Please enter valid numeric values for all fields.", "status": "error"}
        return render_template("heart_dashboard.html", **build_dashboard_context(request.form, prediction=prediction))

    final_data = pd.DataFrame([data], columns=FIELD_NAMES)
    prediction = model.predict(final_data)
    is_positive = int(prediction[0]) == 1
    result = "Positive" if is_positive else "Negative"

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(final_data)[0]
        probability = float(proba[int(prediction[0])] * 100)

    selected_row = request.form.get("row_index") or None
    form_values = {field: request.form.get(field, "") for field in FIELD_NAMES}
    return render_template(
        "heart_dashboard.html",
        **build_dashboard_context(
            form_values=form_values,
            selected_row=selected_row,
            prediction={
                "text": result,
                "status": "risk" if is_positive else "safe",
                "probability": probability,
            },
        ),
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
