"""
Модель в фитнес центр для предсказания ухода клиентов
Использую классификацию (Logistic Regression)
"""
import math
import random

dataset = {
    "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "age": [25, 40, 30, 50, 22, 35, 28, 45, 33, 38],
    "gender": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
    "monthly_fee": [50, 70, 60, 80, 45, 65, 55, 75, 60, 68],
    "tenure_months": [12, 24, 8, 36, 5, 18, 10, 30, 14, 20],
    "support_calls": [1, 3, 0, 5, 2, 1, 4, 3, 0, 2],
    "churned": [0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
}
users_id = dataset.pop("customer_id")
y = dataset.pop("churned")


def normalize_data(data):
    new_x = []
    for lst in data:
        r = []
        max_value = max(lst)
        min_value = min(lst)
        for value in lst:
            r.append((value - min_value) / (max_value - min_value))
        new_x.append(r)
    return new_x


def edit_gender_data(genders):
    """Создает ключи F, M и заполняет в зависимости от гендера 0/1"""
    male = []
    female = []

    for gender in genders:
        if gender == "F":
            male.append(0)
            female.append(1)
        else:
            male.append(1)
            female.append(0)

    dataset["male"] = male
    dataset["female"] = female


def log_loss(y, p):
    """
    функция ошибки
    """
    eps = 1e-15
    p = min(max(p, eps), 1 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict_proba(X, w, b):
    z = b
    for i in range(len(w)):
        z += X[i] * w[i]
    return sigmoid(z)


def gradient(X, y, w, b):
    """
    Поиск градиента
    """
    n = len(w)
    m = len(X[0])
    dw = [0] * m
    db = 0

    for i in range(n):
        p = predict_proba(X[i], w, b)
        error = p - y[i]

        for j in range(m):
            dw[j] += error * X[i][j]

        db += error

    return [value / n for value in dw], db


def train(X, y, lr=0.1, epochs=1000):
    """
    Функция обучения
    """
    w = [random.random() for _ in range(len(X[0]))]
    b = 0

    for epoch in range(epochs):
        dw, db = gradient(X, y, w, b)

        for j in range(len(w)):
            w[j] -= dw[j] * lr
        b -= db * lr

        if epoch % 50 == 0:
            loss = 0
            for k in range(len(y)):
                p = predict_proba(X[k], w, b)
                loss += log_loss(y[k], p)
            print(f"epoch {epoch}: loss={loss/len(X):.4f}")


    return w, b


edit_gender_data(dataset["gender"])
features = ["age", "monthly_fee", "tenure_months", "support_calls", "male", "female"]
X = []
for i in range(len(y)):
    row = []
    for feat in features:
        row.append(dataset[feat][i])
    X.append(row)


w, b = train(normalize_data(X), y, epochs=3000)
