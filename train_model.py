
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = {
    "hour": [7, 8, 9, 10, 12, 14, 16, 17, 18, 19,
             20, 21, 8, 9, 17, 18, 11, 13, 16, 19],
    "day_type": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    "vehicles": [40, 90, 100, 60, 50, 40, 80, 100, 110, 90,
                 50, 30, 35, 40, 45, 50, 30, 35, 85, 95],
    "congestion": ["Low", "High", "High", "Medium", "Low",
                   "Low", "Medium", "High", "High", "Medium",
                   "Low", "Low", "Low", "Low", "Low", "Medium",
                   "Low", "Low", "Medium", "High"]
}

df = pd.DataFrame(data)

X = df[["hour", "day_type", "vehicles"]]
y = df["congestion"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("traffic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training completed!")
print("Model saved as traffic_model.pkl")
