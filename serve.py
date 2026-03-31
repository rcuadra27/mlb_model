from flask import Flask, request, jsonify
import lightgbm as lgb
import numpy as np
import pandas as pd

app = Flask(__name__)

FEATURES = open("/model/runs_model_v8_features.txt").read().splitlines()
model = lgb.Booster(model_file="/model/runs_model_v8.txt")

TRAINING_LEAGUE_MEAN = 4.50

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = pd.DataFrame(data["instances"], columns=FEATURES)
    league_baseline = np.array(data.get("league_baseline", [TRAINING_LEAGUE_MEAN] * len(X)))
    residuals = model.predict(X).astype(float)
    predicted_runs = residuals + league_baseline
    return jsonify({"predictions": predicted_runs.tolist()})

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
