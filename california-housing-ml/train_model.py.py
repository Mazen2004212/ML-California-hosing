# ========== Step 1: استيراد المكتبات ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="X has feature names", category=UserWarning)

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import joblib

# ========== Step 2: تحميل البيانات ==========
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# ========== Step 3: معالجة البيانات ==========
df['Population'] = np.log1p(df['Population'])
df['AveOccup'] = np.log1p(df['AveOccup'])
df['RoomsPerPerson'] = df['AveRooms'] / (df['Population'] + 1e-5)
df['BedsPerRoom'] = df['AveBedrms'] / (df['AveRooms'] + 1e-5)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ========== Step 4: تجهيز البيانات ==========
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude',
            'RoomsPerPerson', 'BedsPerRoom']
X = df[features]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Step 5: بناء الـ Pipelines باستخدام MinMaxScaler ==========
pipelines = {
    'linear': Pipeline([('scaler', MinMaxScaler()), ('model', LinearRegression())]),
    'knn': Pipeline([('scaler', MinMaxScaler()), ('model', KNeighborsRegressor(n_neighbors=5, weights="distance"))]),
    'xgb': Pipeline([('scaler', MinMaxScaler()), ('model', XGBRegressor(objective='reg:squarederror', random_state=42))])
}

# ========== Step 6: تدريب الموديلات ==========
results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'mse': mse, 'r2': r2}
    joblib.dump(pipe, f"{name}_model.pkl")
    print(f"\n📊 {name.upper()} - MSE: {mse:.4f} | R²: {r2:.4f}")

# ========== Step 7: رسم المقارنة ==========
model_names = list(results.keys())
mse_scores = [results[n]['mse'] for n in model_names]
r2_scores = [results[n]['r2'] for n in model_names]

plt.figure(figsize=(10, 5))
plt.bar(model_names, mse_scores)
plt.title("Model Comparison - MSE")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("model_comparison_mse.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(model_names, r2_scores)
plt.title("Model Comparison - R²")
plt.ylabel("R² Score")
plt.tight_layout()
plt.savefig("model_comparison_r2.png")
plt.show()

# ========== Step 8: اختبار التوقع اليدوي ==========
print("\n🔍 Sanity Check:")

def build_input(sample):
    return pd.DataFrame([{
        'MedInc': sample[0], 'HouseAge': sample[1], 'AveRooms': sample[2], 'AveBedrms': sample[3],
        'Population': sample[4], 'AveOccup': sample[5], 'Latitude': sample[6], 'Longitude': sample[7],
        'RoomsPerPerson': sample[2] / (sample[4] + 1e-5),
        'BedsPerRoom': sample[3] / (sample[2] + 1e-5)
    }])[features]

sample1 = build_input([3, 15, 4, 1, 8000, 2, 34.5, -120])
sample2 = build_input([10, 1, 20, 2, 50, 0.5, 39.0, -115])

for name in pipelines:
    model = joblib.load(f"{name}_model.pkl")
    scaler = model.named_steps['scaler']
    scaled1 = scaler.transform(sample1)
    scaled2 = scaler.transform(sample2)

    print(f"\n{name.upper()} Scaled Sample 1:\n{scaled1}")
    print(f"{name.upper()} Scaled Sample 2:\n{scaled2}")
    print(f"🔁 Distance between samples: {np.linalg.norm(scaled1 - scaled2):.4f}")

    pred1 = model.predict(sample1)[0]
    pred2 = model.predict(sample2)[0]
    print(f"🔮 {name.upper()} Prediction 1: {pred1:.4f}, Prediction 2: {pred2:.4f}, Δ = {abs(pred1 - pred2):.4f}")
