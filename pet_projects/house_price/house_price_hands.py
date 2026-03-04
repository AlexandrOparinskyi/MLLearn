"""
Предсказание цены дома
Модель Linear Regression
"""
import random

dataset = {
    "area": [80, 120, 60, 150, 200, 90, 110, 70, 160, 130],
    "bedrooms": [2, 3, 1, 4, 5, 2, 3, 2, 4, 3],
    "bathrooms": [1, 2, 1, 2, 3, 1, 2, 1, 3, 2],
    "floors": [1, 1, 1, 2, 2, 1, 1, 1, 2, 2],
    "year_built": [2005, 2010, 1995, 2018, 2020, 2000, 2008, 1990, 2015, 2012],
    "distance_to_city": [12, 8, 15, 5, 3, 10, 7, 18, 6, 9],
    "price": [120000, 180000, 95000, 260000, 350000, 135000, 170000, 85000, 280000, 210000]
}
y = dataset.pop("price")
features = list(dataset.keys())


def normalize_param(param: list[int]):
    """
    Из-за сильного разброса данных используем нормализацию
    """
    new_param = []
    max_param = max(param)
    min_param = min(param)
    for p in param:
        new_param.append((p - min_param) / (max_param - min_param))
    return new_param

norm_dataset = {}
for key, value in dataset.items():
    norm_dataset[key] = normalize_param(value)


X = []
for i in range(len(y)):
    row = []
    for feat in features:
        row.append(norm_dataset[feat][i])
    X.append(row)


def predict(X, w, b):
    y_pred = 0
    for i in range(len(w)):
        y_pred += X[i] * w[i]
    return y_pred + b


def error_mse(X, y, w, b):
    error = 0
    for i in range(len(X)):
        y_pred = predict(X[i], w, b)
        error += (y[i] - y_pred) ** 2
    return (1 / len(X)) * error


def gradient(X, y, w, b):
    n = len(w)
    m = len(X)
    dw = [0] * len(w)
    db = 0

    for i in range(m):
        y_pred = predict(X[i], w, b)
        error = y_pred - y[i]

        for j in range(n):
            dw[j] += error * X[i][j]
        db += error

    return [val / m for val in dw], db / m


def train(X, y, lr=0.1, epochs=3000):
    w = [random.random() for _ in range(len(X[0]))]
    b = 0

    for epoch in range(epochs):
        dw, db = gradient(X, y, w, b)

        for j in range(len(dw)):
            w[j] -= dw[j] * lr
        b -= db * lr

        if epoch % 50 == 0:
            error = error_mse(X, y, w, b)
            print(error)

    return w, b


w, b = train(X, y)

