# 🏁 F1 Las Vegas Grand Prix Winner Prediction

Predicting the winner of the LAS VEGAS F1 Grand Prix using real-world data and machine learning!

---

## 🚀 Live Demo

🌐 [**Website Link**](https://f1-winner-predictor.netlify.app/)

---

## 🎯 Objective

**Build a model or workflow to predict the winner of the upcoming Las Vegas F1 Grand Prix.**

- Utilize historical race data, driver/team statistics, qualifying performances, track characteristics, and external factors (e.g., weather, news).
- Provide code (notebooks/scripts), explain key insights, discuss uncertainty and modeling limitations, and justify results.

---

## 📊 Approach & Workflow

1. **Data Collection**
    - Gathered F1 historical race results, driver/team stats, qualifying results, track info, weather, and contextual news.
2. **Feature Engineering & Selection**
    - Selected relevant features:
        - Driver skill/history, team form, qualifying performance
        - Las Vegas circuit properties, previous street track results
        - Weather (if available), recent developments
3. **Modeling**
    - Explored various modeling approaches:
        - Logistic Regression
        - Ensemble models (Random Forest/XGBoost)
        - Probability calibration for winner prediction
    - Model selection based on metrics (accuracy, log-loss, calibration).
4. **Prediction & Inference**
    - Ran model to output winner probabilities for all drivers in the Las Vegas race.
5. **Result Interpretation**
    - Discussed model confidence, edge cases, and scenarios for pre/post-race changes.

---

## 💡 Key Insights

- Data shows that qualifying position and team performance are strong predictors of race outcome.
- Street circuits like Las Vegas introduce more uncertainty compared to traditional tracks.
- External news (driver lineup changes, upgrades) can shift win probabilities last minute.

---

## 📈 Results

- The following table shows the top predicted winners and their probabilities:

| Position | Driver           | Probability (%) |
|----------|------------------|----------------|
| 1        | Oscar Piastri    | 73.0           |
| 2        | Lando Norris     | 70.0           |
| 3        | Max Verstappen   | 57.0           |
| ...      | ...              | ...            |

> *See website for Live, interactive predictions and data visualizations.*

---

## ⚠️ Limitations & Uncertainty

- Model trained on available public data; real-world results may differ due to unforeseen factors.
- Predictive accuracy on new or unique circuits is inherently lower.
- External shocks (weather, crashes, strategy calls) impact post-race outcomes.

---

## 📁 Repository Structure
F1 Predictor/
├── backend/ # Flask API and model code
├── frontend/ # Flutter web app (UI)
├── data/ # CSVs for drivers & predictions
├── app.py # Backend entry point
├── README.md # Project documentation (you are here!)
└── .gitignore

🛠️ Usage

1. **Clone this repo:**  
git clone https://github.com/DevKumar005/Formula-1-Winner-Predictor

2. **Backend Setup (Python/Flask):**
- Navigate to `backend`, install requirements, and run `app.py`.

3. **Frontend Setup (Flutter):**
- Navigate to `frontend`, run `flutter pub get`, then `flutter run -d web-server`.
- Update API URLs if deploying.

4. **Live Demo:**  
- Visit the deployed website (link above) for real-time predictions.

---

## 📝 Deliverables

- Source code (Python notebooks/scripts & Flutter app)
- Trained model and data processing scripts
- Key insights and visualizations (see website)
- Complete deployment for web app prediction service

---

## 🤝 Contributing

Contributions and feedback are welcome!  
Feel free to open issues or submit pull requests.

---

## 📝 License

MIT License.  
*See LICENSE file for details.*
