import sys
import os
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFormLayout, QLineEdit, QPushButton,
    QLabel, QMessageBox, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QComboBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# تحميل الموديلات
model_lin = joblib.load("linear_model.pkl")
model_knn = joblib.load("knn_model.pkl")
model_xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

input_features = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]
full_features = input_features + ['RoomsPerPerson', 'BedsPerRoom']

class HousePriceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏠 California House Price Predictor")
        self.setGeometry(200, 200, 1200, 600)
        self.setStyleSheet("background-color: #e6f2ff; font-family: Arial;")
        self.inputs = {}
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        form_group = QGroupBox("🔢 Input Features")
        form_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; color: #003366; }")
        form_layout = QFormLayout()

        default_values = {
            'MedInc': '5.0', 'HouseAge': '30', 'AveRooms': '6',
            'AveBedrms': '1', 'Population': '1500', 'AveOccup': '3',
            'Latitude': '36', 'Longitude': '-120'
        }

        for feature in input_features:
            label = QLabel(feature)
            label.setFont(QFont("Arial", 10, QFont.Bold))
            entry = QLineEdit()
            entry.setText(default_values[feature])
            entry.setFont(QFont("Arial", 10))
            self.inputs[feature] = entry
            form_layout.addRow(label, entry)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Linear Regression", "KNN Regressor", "XGBoost Regressor"])
        form_layout.addRow(QLabel("Select Model:"), self.model_selector)

        buttons = [
            ("🔮 Predict Price", self.predict_price, "#007acc"),
            ("📈 Show Line Chart", self.show_prediction_chart, "#28a745"),
            ("📦 Show Boxplot", self.show_boxplot_chart, "#ffc107"),
            ("📊 Correlation Heatmap", self.show_heatmap, "#ff6600"),
            ("📉 Feature Histograms", self.show_histograms, "#cc33ff"),
            ("📦 Feature Boxplots", self.show_feature_boxplots, "#339999"),
            ("🎯 Actual vs Predicted", self.show_actual_vs_predicted, "#9933cc"),
            ("🗑️ Clear History", self.clear_history, "#cc0000")
        ]

        for text, action, color in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; padding: 8px;")
            btn.clicked.connect(action)
            form_layout.addRow(btn)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group, 1)

        self.table = QTableWidget()
        self.table.setColumnCount(len(input_features) + 2)
        self.table.setHorizontalHeaderLabels(input_features + ['Predicted Price', 'Timestamp'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 2)

        self.setLayout(layout)
        self.load_predictions()

    def predict_price(self):
        try:
            values = {feat: float(self.inputs[feat].text()) for feat in input_features}
            values['RoomsPerPerson'] = values['AveRooms'] / (values['Population'] + 1e-5)
            values['BedsPerRoom'] = values['AveBedrms'] / (values['AveRooms'] + 1e-5)

            df = pd.DataFrame([values])[full_features]
            scaled = scaler.transform(df)

            selected_model = self.model_selector.currentText()
            if selected_model == "Linear Regression":
                prediction = model_lin.predict(scaled)[0]
            elif selected_model == "KNN Regressor":
                prediction = model_knn.predict(scaled)[0]
            elif selected_model == "XGBoost Regressor":
                prediction = model_xgb.predict(scaled)[0]
            else:
                raise Exception("Model not recognized")

            price = round(prediction * 100000, 2)

            values_to_save = {feat: values[feat] for feat in input_features}
            values_to_save['Predicted Price'] = price
            values_to_save['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.save_prediction(values_to_save)

            QMessageBox.information(self, "Prediction", f"🏡 Estimated Price: ${price}")
            self.load_predictions()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Error: {str(e)}")

    def save_prediction(self, row_data):
        df = pd.DataFrame([row_data])
        if os.path.exists("prediction_log.csv"):
            existing = pd.read_csv("prediction_log.csv")
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv("prediction_log.csv", index=False)

    def load_predictions(self):
        if os.path.exists("prediction_log.csv"):
            df = pd.read_csv("prediction_log.csv")
            self.table.setRowCount(len(df))
            for row in range(len(df)):
                for col in range(len(df.columns)):
                    item = QTableWidgetItem(str(df.iloc[row, col]))
                    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    self.table.setItem(row, col, item)

    def show_prediction_chart(self):
        try:
            df = pd.read_csv("prediction_log.csv")
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.sort_values("Timestamp")
            plt.figure(figsize=(10, 5))
            plt.plot(df["Timestamp"], df["Predicted Price"], marker='o', linestyle='-', color='blue')
            plt.title("📈 Predicted House Prices Over Time")
            plt.xlabel("Timestamp")
            plt.ylabel("Predicted Price ($)")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Chart Error: {str(e)}")

    def show_boxplot_chart(self):
        try:
            df = pd.read_csv("prediction_log.csv")
            plt.figure(figsize=(6, 5))
            plt.boxplot(df["Predicted Price"], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#1f77b4", color="#1f77b4"),
                        medianprops=dict(color="white"))
            plt.title("📦 Distribution of Predicted House Prices")
            plt.ylabel("Price ($)")
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Boxplot Error: {str(e)}")

    def show_heatmap(self):
        try:
            df = pd.read_csv("prediction_log.csv")
            numeric_df = df.select_dtypes(include=[np.number])
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("📊 Correlation Heatmap")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Heatmap Error: {str(e)}")

    def show_histograms(self):
        try:
            df = pd.read_csv("prediction_log.csv")
            df.hist(figsize=(15, 10), bins=30)
            plt.suptitle("📉 Feature Histograms", fontsize=16)
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Histogram Error: {str(e)}")

    def show_feature_boxplots(self):
        try:
            df = pd.read_csv("prediction_log.csv")
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=df)
            plt.title("📦 Feature Boxplots")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Boxplot Error: {str(e)}")

    def show_actual_vs_predicted(self):
        try:
            df = pd.read_csv("prediction_log.csv")
            df["Index"] = range(len(df))
            plt.figure(figsize=(8, 5))
            plt.plot(df["Index"], df["Predicted Price"], label="Predicted", color="blue")
            plt.xlabel("Prediction Index")
            plt.ylabel("Predicted Price")
            plt.title("🎯 Actual vs Predicted (Simulated)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Comparison Chart Error: {str(e)}")

    def clear_history(self):
        if os.path.exists("prediction_log.csv"):
            os.remove("prediction_log.csv")
            QMessageBox.information(self, "Cleared", "✅ Prediction history has been cleared.")
            self.load_predictions()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HousePriceApp()
    window.show()
    sys.exit(app.exec_())
