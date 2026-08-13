import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = pd.read_csv("data/car data.csv")

print("FIRST 5 ROWS:")
print(data.head())

print("\nDATASET SIZE:")
print(data.shape)

print("\nCOLUMN NAMES:")
print(data.columns)

print("\nDATASET INFORMATION:")
data.info()

print("\nMISSING VALUES:")
print(data.isnull().sum())

print("\nSTATISTICAL SUMMARY:")
print(data.describe())

print("\nFUEL TYPES:")
print(data["Fuel_Type"].value_counts())

print("\nSELLER TYPES:")
print(data["Seller_Type"].value_counts())

print("\nTRANSMISSION TYPES:")
print(data["Transmission"].value_counts())

print("\nNUMBER OF OWNERS:")
print(data["Owner"].value_counts())

print("\nPREPARING DATA FOR MACHINE LEARNING")

X = data.drop(["Selling_Price", "Car_Name"], axis=1)

y = data["Selling_Price"]

print("\nINPUT FEATURES (X):")
print(X.head())

print("\nTARGET (y):")
print(y.head())

print("\nX SHAPE:")
print(X.shape)

print("\ny SHAPE:")
print(y.shape)

print("\nENCODING CATEGORICAL DATA")

X_encoded = pd.get_dummies(
    X,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

print("\nENCODED DATA:")
print(X_encoded.head())

print("\nENCODED SHAPE:")
print(X_encoded.shape)

print("\nSPLITTING DATA")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTESTING DATA:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nCREATING MACHINE LEARNING MODEL")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("\nTRAINING THE MODEL...")

model.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED!")

import joblib

joblib.dump(model, "car_price_model.pkl")
joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")

print("MODEL SAVED SUCCESSFULLY!")



print("\nMAKING PREDICTIONS")

predictions = model.predict(X_test)

print("Predictions:")
print(predictions[:10])

print("\nMODEL EVALUATION")

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

print("\nACTUAL VS PREDICTED PRICES")

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print(comparison.head(10))

print("\n" + "=" * 50)
print("       USED CAR PRICE PREDICTOR")
print("=" * 50)

year = int(input("Enter car manufacturing year: "))

present_price = float(input("Enter present price (in lakh): "))

kms_driven = int(input("Enter kilometers driven: "))

fuel_type = input("Enter fuel type (Petrol/Diesel/CNG): ")

seller_type = input("Enter seller type (Dealer/Individual): ")

transmission = input("Enter transmission (Manual/Automatic): ")

owner = int(input("Enter number of previous owners (0/1/2/3): "))


car = pd.DataFrame({
    "Year": [year],
    "Present_Price": [present_price],
    "Kms_Driven": [kms_driven],
    "Fuel_Type": [fuel_type],
    "Seller_Type": [seller_type],
    "Transmission": [transmission],
    "Owner": [owner]
})


car_encoded = pd.get_dummies(
    car,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)


car_encoded = car_encoded.reindex(
    columns=X_encoded.columns,
    fill_value=0
)


predicted_price = model.predict(car_encoded)


print("\n" + "=" * 50)
print("       PREDICTION RESULT")
print("=" * 50)

print(f"Estimated Selling Price: ₹{predicted_price[0]:.2f} lakh")
print(f"Approximately: ₹{predicted_price[0] * 100000:,.0f}")

print("=" * 50)

print("\n" + "=" * 50)
print("       FEATURE IMPORTANCE")
print("=" * 50)

importance = model.feature_importances_

feature_names = X_encoded.columns

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Used Car Price Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Used Car Price Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()
