import os
import random

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

import model as M

app = Flask(__name__)
app.secret_key = "change-me"
CSV = os.path.join(M.BASE, "creditcard.csv")

clf, metrics = M.load()


def ensure_model():
    """Train automatically on first start if creditcard.csv is present."""
    global clf, metrics
    if clf is None and os.path.exists(CSV):
        metrics = M.train(CSV)
        clf, metrics = M.load()


ensure_model()


@app.route("/")
def index():
    return render_template("index.html", m=metrics, csv_ok=os.path.exists(CSV))


@app.route("/train", methods=["POST"])
def train():
    global clf, metrics
    if not os.path.exists(CSV):
        flash("creditcard.csv not found next to app.py.", "err")
        return redirect(url_for("index"))
    M.train(CSV)
    clf, metrics = M.load()
    flash("Model retrained on a fresh 70/30 split.", "ok")
    return redirect(url_for("index"))


def parse_row(form):
    vals = {}
    for f in M.FEATURES:
        raw = (form.get(f) or "").strip()
        if raw == "":
            raise ValueError(f"'{f}' is empty.")
        try:
            vals[f] = float(raw)
        except ValueError:
            raise ValueError(f"'{f}' must be a number.")
    if not 0 <= vals["Time"] <= M.TIME_MAX:
        raise ValueError(f"Time must be between 0 and {M.TIME_MAX} seconds.")
    if vals["Amount"] < 0:
        raise ValueError("Amount cannot be negative.")
    return vals


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, values, error = None, {}, None
    if request.method == "POST":
        values = {f: request.form.get(f, "") for f in M.FEATURES}
        if clf is None:
            error = "No trained model yet. Train it on the dashboard first."
        else:
            try:
                row = parse_row(request.form)
                prob = float(clf.predict_proba(pd.DataFrame([row])[M.FEATURES])[0, 1])
                result = {"prob": prob, "pct": round(prob * 100, 2),
                          "fraud": prob >= M.THRESHOLD}
            except ValueError as e:
                error = str(e)
    return render_template("predict.html", features=M.FEATURES, values=values,
                           result=result, error=error, tmax=M.TIME_MAX,
                           trained=clf is not None)


@app.route("/sample")
def sample():
    """Return a random real example from the 30% test set."""
    cls = int(request.args.get("cls", 0))
    if not os.path.exists(M.SAMPLES_PATH):
        return jsonify({"error": "train first"}), 404
    df = pd.read_csv(M.SAMPLES_PATH)
    df = df[df.Class == cls]
    if df.empty:
        return jsonify({"error": "no samples"}), 404
    row = df.iloc[random.randrange(len(df))]
    return jsonify({f: float(row[f]) for f in M.FEATURES})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
