import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("../dataset/restaurant_sales.csv")

# Encode categorical variables
label_encoder_day = LabelEncoder()
label_encoder_dish = LabelEncoder()

df["Day_Encoded"] = label_encoder_day.fit_transform(df["Day"])
df["Dish_Encoded"] = label_encoder_dish.fit_transform(df["Dish"])

# Features and target variable
X = df[["Day_Encoded", "Dish_Encoded"]]
y = df["Quantity_Sold"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model RMSE: {rmse:.2f}")

# Save model and encoders
joblib.dump(model, "../backend/models/sales_model.pkl")
joblib.dump(label_encoder_day, "../backend/models/label_encoder_day.pkl")
joblib.dump(label_encoder_dish, "../backend/models/label_encoder_dish.pkl")

print("Model and encoders saved successfully.")
