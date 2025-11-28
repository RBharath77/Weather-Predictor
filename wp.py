import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

file_path = "/mnt/data/Chennai_1990_2022_Madras.csv"
df = pd.read_csv(file_path)

df = df.dropna(subset=['tavg', 'tmin', 'tmax', 'prcp'])
df.fillna(method='ffill', inplace=True)

df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['Year'] = df['time'].dt.year
df['Month'] = df['time'].dt.month

def classify_weather(row):
    if row['prcp'] > 10:
        return 'Rainy'
    elif row['tavg'] < 25:
        return 'Cool'
    elif 25 <= row['tavg'] <= 32:
        return 'Moderate'
    else:
        return 'Hot'

df['Weather'] = df.apply(classify_weather, axis=1)

le = LabelEncoder()
df['Weather_encoded'] = le.fit_transform(df['Weather'])

features = ['tavg', 'tmin', 'tmax', 'prcp', 'Month', 'Year']
X = df[features]
y = df['Weather_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

future_data = df[df['Year'] == 2022].copy()
future_data['Year'] = 2023
X_future = future_data[features]
future_data['Predicted_Weather'] = le.inverse_transform(model.predict(X_future))

print(future_data[['time', 'tavg', 'tmin', 'tmax', 'prcp', 'Predicted_Weather']].head(10))

weather_counts = df['Weather'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Weather Distribution in Chennai (1990–2022)")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[features + ['Weather_encoded']].corr(), annot=True, cmap='mako')
plt.title('Feature Correlation with Weather')
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Month', y='tavg', hue='Weather', palette='tab10')
plt.title('Average Temperature Trend by Month')
plt.xlabel('Month')
plt.ylabel('Average Temperature (°C)')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='Weather', data=df[df['Year'] == 2022], alpha=0.6, label='Actual')
sns.countplot(x='Predicted_Weather', data=future_data, alpha=0.6, label='Predicted')
plt.title('Actual (2022) vs Predicted (2023) Weather Comparison')
plt.legend()
plt.show()
