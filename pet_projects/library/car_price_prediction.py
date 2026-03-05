import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("../../datasets/car_price_pred.csv")
X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
numeric_features = [
    "Engine_Size",
    "Horsepower",
    "Mileage",
    "Age",
    "Doors"
]
categorical_features = [
    "Brand",
    "Fuel_Type",
    "Transmission"
]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
# Обучение
pipeline.fit(X_train, y_train)

# Проверка на тестовых данных
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Логирование предсказания модели
print(f"MAE: {mae:.4f}; MSE: {mse:.4f}; RMSE: {rmse:.4f}' R2: {r2:.4f}")

# Предсказание для новой машины
new_car = pd.DataFrame({
    "Brand": ["Toyota"],
    "Fuel_Type": ["Petrol"],
    "Transmission": ["Manual"],
    "Engine_Size": [1.6],
    "Horsepower": [120],
    "Mileage": [40000],
    "Age": [3],
    "Doors": [4]
})
new_price = pipeline.predict(new_car)
print(f"Predicted price: {new_price[0]:.2f}")
print(f"Price this car in dataset: 15000")
