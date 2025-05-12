# 🏡 California Housing Price Predictor

This project is a web-based machine learning application that predicts California housing prices using multiple regression models. It offers interactive charts, a model comparison interface, and a user-friendly web dashboard.

## ⚙️ Features

- Predict housing prices based on user input
- Supports multiple models:
  - Linear Regression
  - K-Nearest Neighbors (KNN)
  - XGBoost Regressor
- Displays charts and model comparisons (MSE, R²)
- Logs predictions to a CSV file
- Web interface using Flask, HTML, and CSS
- Interactive charts with statistical insights

## 📁 Project Structure

california-housing-ml/
│
├── app.py # Flask web app
├── gui_predictor.py.py # Optional GUI implementation
├── train_model.py.py # Model training script
├── knn_model.pkl # Trained KNN model
├── linear_model.pkl # Trained Linear Regression model
├── xgb_model.pkl # Trained XGBoost model
├── scaler.pkl # Data scaler
├── prediction_log.csv # CSV log of all predictions
│
├── static/
│ ├── prediction_log.csv # Static copy
│ └── charts/ # All generated charts
│ ├── boxplot.png
│ ├── feature_boxplots.png
│ ├── feature_importance.png
│ ├── heatmap.png
│ ├── histograms.png
│ └── line_chart.png
│
├── templates/
│ ├── index.html # Home page
│ └── charts.html # Chart dashboard


## 🚀 Getting Started

1. Clone the repository or extract the zip.
2. Install required dependencies:
```bash
pip install -r requirements.txt

Run the application:

python app.py

Visit http://localhost:5000 in your browser.

📊 Models Used
Linear Regression: Simple baseline regressor

K-Nearest Neighbors (KNN): Distance-based regression

XGBoost Regressor: Gradient-boosted trees for advanced performance

📈 Visualizations
Feature importance (via XGBoost)

Correlation heatmaps

Histograms & Boxplots for feature analysis

Model performance comparisons (MSE and R²)

✅ Requirements
Python 3.x

Flask

scikit-learn

XGBoost

pandas, numpy, matplotlib, seaborn

✨ Developed By
[Mazen ibrahim, Mohamed abd el-gawad, Mohamed ahmed, Hala mazen]

Dataset: [California Housing Dataset from scikit-learn]
