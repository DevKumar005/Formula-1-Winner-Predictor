# 🏁 F1 Las Vegas Grand Prix — Winner Prediction Engine

![Formula 1](https://img.shields.io/badge/Formula_1-E10600?style=for-the-badge&logo=formula1&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Machine learning meets motorsport.** A full-stack predictive analytics system trained on 5+ years of Formula 1 race data to forecast the 2025 Las Vegas Grand Prix winner — powered by ensemble models, real-world telemetry features, and a live Flutter web app.

[🌐 **Live Demo**](https://f1-winner-predictor.netlify.app/)

---

## 🎯 What This Project Does

This project builds an end-to-end ML pipeline that ingests historical F1 race data, engineers predictive features, trains and compares multiple classification models, and exposes real-time winner probabilities through a deployed web application.

The Las Vegas Strip Circuit — a 6.12 km night race through the city's iconic streets — presents a unique forecasting challenge: limited historical data, unpredictable street circuit behavior, and a volatile competitive field. This project addresses that challenge head-on.

---

## 📐 Architecture Overview

```
Raw F1 Data (2020–2025)
        │
        ▼
┌───────────────────┐
│  Data Collection  │  fetch_race_data.py · f1_schedule.py
│  & Combination    │  → 100+ race CSVs · f1_all_races_combined.csv
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Cleaning &       │  clean_data.py · inspect_data.py · explore_data.py
│  Exploration      │  → f1_data_cleaned.csv
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Feature          │  features.py · prepare_ml.py
│  Engineering      │  → f1_features_engineered.csv · X_train/test · scaler.pkl
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Model Training   │  model.py · model_baseline.py · compare_models.py
│  & Comparison     │  → logistic_regression_model.pkl · random_forest_model.pkl
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Las Vegas        │  las_vegas_predict.py · visualize_lv_predictions.py
│  Inference        │  → las_vegas_2025_predictions.csv · win probability charts
└────────┬──────────┘
         │
         ▼
┌───────────────────┐    ┌─────────────────────────┐
│  Flask REST API   │◄───│  Flutter Web Frontend   │
│  (backend/app.py) │    │  (frontend/lib/)        │
└───────────────────┘    └─────────────────────────┘
```

---

## 📊 Data Pipeline

### Coverage
| Season | Races Included |
|--------|---------------|
| 2020   | 17 Grands Prix |
| 2021   | 18 Grands Prix |
| 2022   | 22 Grands Prix |
| 2023   | 21 Grands Prix |
| 2024   | 22 Grands Prix |
| 2025   | 20 Grands Prix (ongoing) |
| **Total** | **~120 races · thousands of driver-race entries** |

### Feature Engineering Highlights

The model learns from a rich feature set across four categories:

- **Driver Performance** — historical win rate, podium rate, points-per-race, recent form (rolling 5-race window)
- **Team / Constructor** — constructor standings position, recent upgrade trajectory, reliability index
- **Qualifying** — grid position, gap to pole (seconds), front-row starts ratio
- **Circuit Context** — street circuit flag, circuit similarity score to Las Vegas layout, overtaking difficulty index

---

## 🤖 Modeling

Three approaches were evaluated and compared:

| Model | Strengths | Key Metric |
|-------|-----------|-----------|
| Logistic Regression (baseline) | Interpretable, fast | Log-loss baseline |
| Random Forest | Handles non-linearity, feature importance | Best calibration |
| XGBoost (explored) | Gradient boosting, robust to noise | Accuracy |

Model selection was driven by **log-loss** and **calibration curves** — ensuring that a 70% predicted probability actually reflects ~70% empirical win frequency. Overconfident models are penalized.

Final models are serialized to:
```
backend/data/logistic_regression_model.pkl
backend/data/random_forest_model.pkl
backend/data/scaler.pkl
backend/data/feature_columns.pkl
```

---

<a id="results"></a>

## 📈 Results — 2025 Las Vegas Grand Prix Predictions

> Probabilities reflect model confidence at time of prediction. Pre-race grid and team updates may shift rankings.

| Predicted Position | Driver | Win Probability |
|-------------------|--------|----------------|
| 🥇 1st | Oscar Piastri | 73.0% |
| 🥈 2nd | Lando Norris | 70.0% |
| 🥉 3rd | Max Verstappen | 57.0% |
| … | … | … |

**Key drivers of these predictions:**
- McLaren's strong street circuit form and 2025 car development pace
- Verstappen's consistent top-3 qualifying rate despite increased midfield competition
- Las Vegas circuit characteristics favouring high-downforce, low-drag setups

See [`backend/data/las_vegas_2025_predictions.csv`](backend/data/las_vegas_2025_predictions.csv) for full driver probability rankings, and the **[live web app](https://f1-winner-predictor.netlify.app/)** for interactive visualizations.

---

## 💡 Key Insights

**1. Qualifying dominance is the single strongest predictor.**
Drivers starting P1–P3 win ~65% of races in the dataset. Grid position outweighs raw pace in predictive power.

**2. Street circuits amplify uncertainty.**
Model confidence drops ~15% on street circuits vs. traditional tracks. Monaco-style layouts produce more variance; Las Vegas, with its long straights, sits between the two extremes.

**3. Constructor momentum matters more than constructor standing.**
A team on a 3-race improvement trajectory outperforms a team with a higher standing but declining form — captured via the rolling constructor index feature.

**4. Recent news can override historical signals.**
Driver swaps, power unit penalties, and last-minute setup changes are logged as binary override features, shifting probabilities up to ±12%.

---

## ⚠️ Limitations & Uncertainty

- Model trained on publicly available F1 data; proprietary telemetry and team-internal strategy data are not included.
- Las Vegas has limited historical race data (inaugural 2023, second edition 2024), reducing circuit-specific calibration.
- Post-qualifying events (crashes, weather, VSC deployment) are inherently unpredictable and not modelled.
- Probabilities are not betting odds — they represent model-estimated likelihood under known conditions at prediction time.

---

<a id="usage"></a>

## 🛠️ Usage

### Prerequisites
- Python 3.9+
- Flutter 3.x
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/DevKumar005/Formula-1-Winner-Predictor
cd Formula-1-Winner-Predictor
```

### 2. Backend Setup (Python / Flask)

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Re-run the full pipeline
cd backend
python fetch_race_data.py      # Fetch latest race data
python combine_data.py          # Merge CSVs
python clean_data.py            # Clean & validate
python features.py              # Engineer features
python prepare_ml.py            # Scale & split
python model.py                 # Train models
python las_vegas_predict.py     # Generate LV predictions

# Start the API server
cd ..
python app.py
```

API will be available at `http://localhost:5000`.

### 3. Frontend Setup (Flutter Web)

```bash
cd frontend
flutter pub get
flutter run -d web-server --web-port 8080
```

Open `http://localhost:8080` in your browser.

> **Deploying?** Update the API base URL in `frontend/lib/` to point to your hosted backend before building for production.

### 4. Live Demo

Visit the deployed application → **[Website Link](https://f1-winner-predictor.netlify.app/)**

---

## 📝 Deliverables

- [x] Full data pipeline (fetch → clean → engineer → train → predict)

- [x] 100+ race CSVs spanning 2020–2025

- [x] Trained & serialized ML models (Logistic Regression, Random Forest)

- [x] Las Vegas 2025 winner probability rankings

- [x] Prediction visualizations (bar charts, probability plots)

- [x] Flask REST API serving live predictions

- [x] Flutter web app with interactive UI

- [x] Netlify deployment configuration

---

## 🤝 Contributing

Contributions, ideas, and feedback are welcome!

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/better-model`)
3. Commit your changes (`git commit -m 'Add XGBoost with Bayesian hyperparameter tuning'`)
4. Push to the branch (`git push origin feature/better-model`)
5. Open a Pull Request

Please open an issue first for major changes so we can discuss the approach.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

Built by [DevKumar005](https://github.com/DevKumar005)

*"In F1, as in machine learning — the margin between winning and losing is measured in milliseconds and basis points."*
