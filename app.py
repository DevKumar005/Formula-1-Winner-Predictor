from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS
CORS(app)


app = Flask(__name__)
CORS(app)

# Replace with your prediction code
def get_model_predictions():
    # Run your model and return a DataFrame or list of dictionaries
    df = pd.read_csv("data/las_vegas_2025_predictions.csv")
    return df[["FullName", "win_probability"]].rename(
        columns={"FullName": "name", "win_probability": "probability"}
    ).to_dict(orient="records")

@app.route('/api/predictions')
def api_predictions():
    results = get_model_predictions()
    return jsonify(results)

@app.route('/api/drivers')
def api_drivers():
    df = pd.read_csv("data/drivers.csv")
    # drivers.csv must have: name, team, points, podiums, profile_img columns
    return jsonify(df.to_dict(orient="records"))

@app.route('/api/race-info')
def api_race_info():
    info = {
        "name": "Las Vegas Grand Prix 2025",
        "circuit_length": "6.12 km",
        "laps": 50,
        "distance": "306 km",
        "track_map": "https://www.formula1.com/content/dam/fom-website/manual/Misc/Track%20maps/LasVegas_Circuit.png",
        "highlights": "The race returned to F1 in 2023 after decades of absence, quickly becoming a fan favorite due to its vibrant atmosphere and night-time setting.",
        "description": "The Las Vegas Grand Prix is a spectacular night race held on the streets of Las Vegas. The circuit combines a high-speed oval section with tight corners on the city streets, presenting unique challenges to drivers and teams.",
    }
    return jsonify(info)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)