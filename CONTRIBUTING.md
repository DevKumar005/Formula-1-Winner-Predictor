# 🤝 Contributing to F1 Las Vegas Grand Prix Winner Predictor

Thank you for your interest in contributing! Whether it's a bug fix, a new model, better visualizations, or documentation improvements — all contributions are appreciated.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Branch Naming](#branch-naming)
- [Commit Messages](#commit-messages)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## 📜 Code of Conduct

Please be respectful and constructive in all interactions. We're here to build something cool together.

---

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/DevKumar005/Formula-1-Winner-Predictor
   cd Formula-1-Winner-Predictor
   ```
3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a branch** for your work (see [Branch Naming](#branch-naming) below)

---

## 🛠️ How to Contribute

### Backend / ML (Python)
- Improvements to feature engineering go in `backend/features.py`
- New models or training logic go in `backend/model.py`
- Las Vegas-specific inference goes in `backend/las_vegas_predict.py`
- New visualizations go in `backend/visualize_*.py`

### Frontend (Flutter)
- All UI code lives under `frontend/lib/`
- Run locally with `flutter run -d web-server --web-port 8080`
- Keep API base URLs configurable — no hardcoded endpoints

### Data
- Do **not** commit raw race CSVs larger than a few MB
- If adding new data sources, document the source in your PR description

---

## 🌿 Branch Naming

Use a consistent naming pattern:

| Type | Pattern | Example |
|------|---------|---------|
| New feature | `feature/short-description` | `feature/xgboost-model` |
| Bug fix | `fix/short-description` | `fix/scaler-mismatch` |
| Documentation | `docs/short-description` | `docs/update-readme` |
| Data / pipeline | `data/short-description` | `data/add-2025-races` |

---

## ✏️ Commit Messages

Follow this simple format:

```
<type>: <short summary>

Optional longer description if needed.
```

**Types:** `feat`, `fix`, `docs`, `data`, `refactor`, `test`, `chore`

**Examples:**
```
feat: add XGBoost model with Bayesian hyperparameter tuning
fix: resolve scaler mismatch between training and inference
docs: update usage instructions for Flutter 3.x
data: add 2025 race CSVs for rounds 15–20
```

---

## 📬 Pull Request Guidelines

1. **Open an issue first** for any significant change so we can align before you invest time coding
2. Keep PRs focused — one feature or fix per PR
3. Update relevant documentation if your change affects behaviour
4. Make sure existing scripts still run without errors before submitting
5. Fill out the PR description with:
   - What changed and why
   - How to test it
   - Any known limitations

---

## 🐛 Reporting Bugs

Open a [GitHub Issue](https://github.com/DevKumar005/Formula-1-Winner-Predictor/issues) and include:

- A clear title and description
- Steps to reproduce
- Expected vs. actual behaviour
- Python / Flutter version and OS
- Relevant error messages or stack traces

---

## 💡 Suggesting Features

Open a [GitHub Issue](https://github.com/DevKumar005/Formula-1-Winner-Predictor/issues) with the label `enhancement` and describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you considered

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
