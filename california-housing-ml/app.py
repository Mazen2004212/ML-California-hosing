import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
os.makedirs('static/charts', exist_ok=True)

models = {
    'linear': joblib.load("linear_model.pkl"),
    'knn': joblib.load("knn_model.pkl"),
    'xgboost': joblib.load("xgb_model.pkl")
}

input_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                  'Population', 'AveOccup', 'Latitude', 'Longitude']
derived_features = ['RoomsPerPerson', 'BedsPerRoom']
full_features = input_features + derived_features
log_columns = full_features + ['Predicted Price', 'Model Used', 'Timestamp']

@app.route('/')
def home():
    default_values = {
        'MedInc': 5.0, 'HouseAge': 30, 'AveRooms': 6,
        'AveBedrms': 1, 'Population': 1500, 'AveOccup': 3,
        'Latitude': 36, 'Longitude': -120,
        'model_choice': 'xgboost'
    }
    return render_template("index.html", default_values=default_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_choice = data.pop('model_choice', 'xgboost')
        data['RoomsPerPerson'] = data['AveRooms'] / (data['Population'] + 1e-5)
        data['BedsPerRoom'] = data['AveBedrms'] / (data['AveRooms'] + 1e-5)

        df = pd.DataFrame([data])[full_features]
        model = models.get(model_choice)
        prediction = model.predict(df)[0]
        price = round(prediction * 100000, 2)

        log_row = {col: data.get(col, None) for col in full_features}
        log_row['Predicted Price'] = price
        log_row['Model Used'] = model_choice
        log_row['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        log_df = pd.DataFrame([log_row])[log_columns]
        log_path = "static/prediction_log.csv"
        log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

        generate_all_charts()
        return jsonify({'predicted_price': float(price)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/charts')
def charts():
    model_filter = request.args.get("model")
    prediction_count = 0
    last_updated = "N/A"
    model_options = ["linear", "knn", "xgboost"]

    try:
        df = pd.read_csv("static/prediction_log.csv", on_bad_lines='skip')
        if model_filter in model_options:
            df = df[df["Model Used"] == model_filter]

        prediction_count = len(df)
        if "Timestamp" in df.columns and not df.empty:
            last_updated = df["Timestamp"].iloc[-1]

        table_data = df.to_dict(orient="records") if not df.empty else []
        avg_prices = df.groupby("Model Used")["Predicted Price"].mean().to_dict()
    except:
        df = pd.DataFrame(columns=["Timestamp", "Model Used", "Predicted Price"])
        table_data = []
        avg_prices = {}

    return render_template("charts.html",
                           prediction_count=prediction_count,
                           last_updated=last_updated,
                           model_filter=model_filter or "all",
                           model_options=model_options,
                           table_data=table_data,
                           avg_prices=avg_prices)

@app.route('/download-log')
def download_log():
    return send_from_directory('static', "prediction_log.csv", as_attachment=True)

@app.route('/reset-log', methods=['POST'])
def reset_log():
    try:
        log_path = "static/prediction_log.csv"
        if os.path.exists(log_path):
            df = pd.read_csv(log_path, on_bad_lines='skip')
            df.head(0).to_csv(log_path, index=False)
        return jsonify({"message": "✅ Prediction log has been reset."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats')
def stats_page():
    try:
        df = pd.read_csv("static/prediction_log.csv", on_bad_lines='skip')
        total = len(df)
        most_used = df['Model Used'].mode()[0]
        avg_price = df['Predicted Price'].mean()
    except:
        total = 0
        most_used = "N/A"
        avg_price = 0.0

    return render_template("stats.html",
                           total=total,
                           most_used=most_used,
                           avg_price=round(avg_price, 2))

def generate_all_charts():
    log_path = "static/prediction_log.csv"
    if not os.path.exists(log_path):
        return
    try:
        df = pd.read_csv(log_path, on_bad_lines='skip')
        df = df[log_columns].copy()
        df[full_features + ['Predicted Price']] = df[full_features + ['Predicted Price']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        numeric_df = df.select_dtypes(include=[float, int])

        # Boxplot
        plt.figure()
        plt.boxplot(df["Predicted Price"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#1f77b4"), medianprops=dict(color="white"))
        plt.title("Boxplot of Predicted Prices")
        plt.ylabel("Price ($)")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig("static/charts/boxplot.png")
        plt.close()

        # Heatmap
        if numeric_df.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig("static/charts/heatmap.png")
            plt.close()

        # Histograms
        numeric_df.hist(figsize=(15, 10), bins=30)
        plt.suptitle("Feature Histograms")
        plt.tight_layout()
        plt.savefig("static/charts/histograms.png")
        plt.close()

        # Boxplots
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=numeric_df)
        plt.title("Feature Boxplots")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/charts/feature_boxplots.png")
        plt.close()

        # Feature Importance
        model = models["xgboost"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": full_features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=importance_df)
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig("static/charts/feature_importance.png")
            plt.close()
    except Exception as e:
        print(f"❌ Failed to generate charts: {e}")

@app.route('/models')
def model_comparison():
    try:
        # Load evaluation results from saved file or recompute (simplified here)
        results = {
            "linear": {"mse": 0.4631, "r2": 0.6466},
            "knn": {"mse": 0.3469, "r2": 0.7352},
            "xgboost": {"mse": 0.2254, "r2": 0.8280}
        }
    except:
        results = {}

    return render_template("models.html", results=results)

if __name__ == '__main__':
    app.run(debug=True)
