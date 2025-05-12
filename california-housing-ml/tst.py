import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances

# تحميل البيانات
data = fetch_california_housing(as_frame=True)
df = data.frame

# معالجة البيانات
df['Population'] = np.log1p(df['Population'])
df['AveOccup'] = np.log1p(df['AveOccup'])
df['RoomsPerPerson'] = df['AveRooms'] / (df['Population'] + 1e-5)
df['BedsPerRoom'] = df['AveBedrms'] / (df['AveRooms'] + 1e-5)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# الميزات
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude',
            'RoomsPerPerson', 'BedsPerRoom']
X = df[features]
y = df['MedHouseVal']

# تقييس
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسيم
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# تدريب KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# دالة تجهيز العينة
def prepare_sample(sample):
    sample_dict = {
        'MedInc': sample[0], 'HouseAge': sample[1], 'AveRooms': sample[2], 'AveBedrms': sample[3],
        'Population': sample[4], 'AveOccup': sample[5],
        'Latitude': sample[6], 'Longitude': sample[7]
    }
    sample_dict['RoomsPerPerson'] = sample_dict['AveRooms'] / (sample_dict['Population'] + 1e-5)
    sample_dict['BedsPerRoom'] = sample_dict['AveBedrms'] / (sample_dict['AveRooms'] + 1e-5)
    df_sample = pd.DataFrame([sample_dict])[features]
    return scaler.transform(df_sample)

# عينات الاختبار
sample1 = [5, 30, 6, 1, 1500, 3, 36, -120]
sample2 = [9, 5, 30, 10, 100, 0.2, 34, -118]

# تجهيز العينة الأولى
s1 = prepare_sample(sample1)
p1 = knn.predict(s1)[0]

print(f"\n🔮 KNN Prediction for Sample 1: {p1:.4f}")
print(f"\n🔬 Scaled Sample 1:\n{s1}")

# حساب المسافات
dists = pairwise_distances(s1, X_train)
nearest_indices = np.argsort(dists[0])[:5]

print("\n📍 Nearest Neighbors (Top 5):")
for idx in nearest_indices:
    original_index = y_train.index[idx]
    print(f"Index: {idx}, Distance: {dists[0][idx]:.4f}, Target Value: {y_train.iloc[idx]:.4f}")

# مقارنة مع عينة ثانية
s2 = prepare_sample(sample2)
p2 = knn.predict(s2)[0]

print(f"\n🔮 KNN Prediction for Sample 2: {p2:.4f}")
print(f"📏 Distance between Sample 1 and Sample 2: {np.linalg.norm(s1 - s2):.4f}")
print(f"🧮 Prediction Difference: {abs(p1 - p2):.4f}")
