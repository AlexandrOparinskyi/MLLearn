import csv
import math
import random


def read_csv(csv_file: str) -> tuple[dict, list]:
    dataset = {}
    features = {}
    with open(csv_file, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for num, row in enumerate(csv_reader):
            for v_num, value in enumerate(row):
                if num == 0:
                    dataset[value] = []
                    features[v_num] = value
                else:
                    try:
                        dataset[features[v_num]].append(int(value))
                    except ValueError:
                        try:
                            dataset[features[v_num]].append(float(value))
                        except ValueError:
                            dataset[features[v_num]].append(value)

    return dataset, list(features.values())


def one_hot_encoding(data: list) -> dict:
    unique = set(data)
    new_data = {}
    for key in unique:
        new_data[key] = []
    for value in data:
        for key in new_data.keys():
            if key == value:
                new_data[key].append(1)
            else:
                new_data[key].append(0)
    return new_data


def norm_min_max(data: list) -> list:
    norm_data = []
    min_value = min(data)
    max_value = max(data)
    for value in data:
        try:
            norm_data.append((value - min_value) / (max_value - min_value))
        except ZeroDivisionError:
            norm_data.append(0)
    return norm_data


def z_standardize(data):
    mean = sum(data) / len(data)
    stdev = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

    if stdev == 0:
        return [0.0 for _ in range(len(data))]

    return [(x - mean) / stdev for x in data]

def create_matrix(data: dict, feats: list, num: int) -> list:
    matrix = []
    for i in range(num):
        row = []
        for feat in feats:
            row.append(data[feat][i])
        matrix.append(row)
    return matrix


def predict(X, w, b):
    y_pred = 0
    for i in range(len(w)):
        y_pred += X[i] * w[i]
    return y_pred + b


def compute_loss(X, y, w, b):
    error = 0
    for i in range(len(X)):
        y_pred = predict(X[i], w, b)
        error += (y[i] - y_pred) ** 2
    return (1 / len(X)) * error


def gradient(X, y, w, b):
    length_w = len(w)
    length_X = len(X)
    dw = [0] * length_w
    db = 0

    for i in range(length_X):
        y_pred = predict(X[i], w, b)
        error = y_pred - y[i]

        for j in range(length_w):
            dw[j] += error * X[i][j]

        db += error

    return [val / length_X for val in dw], db / length_X


def train(X, y, lr=0.01, epochs=3000):
    w = [random.uniform(-0.01, 0.01) for _ in range(len(X[0]))]
    b = 0

    for epoch in range(epochs):
        dw, db = gradient(X, y, w, b)

        for i in range(len(w)):
            w[i] -= dw[i] * lr
        b -= db * lr

        if epoch % 50 == 0:
            mse = compute_loss(X, y, w, b)
            weights = []
            for weight in w:
                weights.append(f"{weight:.4f}")
            print(f"Epoch: {epoch}, weights: {weights}, "
                  f"bias: {b:.4f}, mse: {mse:.4f}")

    return w, b



def main() -> None:
    dataset, features = read_csv("../../datasets/car_price_pred.csv")
    updated_data = {}
    deleted_keys = set()
    for key, value in dataset.items():
        if isinstance(value[0], str):
            updated_data.update(one_hot_encoding(dataset[key]))
            deleted_keys.add(key)

    for key in deleted_keys:
        if key in dataset:
            dataset.pop(key)
        if key in features:
            features.pop(features.index(key))

    dataset.update(updated_data)
    norm_dataset = {}
    for key, value in dataset.items():
        norm_dataset[key] = z_standardize(value)
    y = dataset.pop("Price")
    X = create_matrix(norm_dataset,
                      features[:-1] + list(updated_data.keys()),
                      len(y))
    w, b = train(X, y)
    # new_car = [-1.010460655954173, -0.9568902828709857, -0.4321041403622522, -0.44174474528453167, 0.0, -0.4472135954999579, -0.4472135954999579, -0.4472135954999579, -0.4472135954999579, -0.4472135954999579, 2.23606797749979, -0.3333333333333333, 0.9354143466934853, -0.7608859102526822, -1.0690449676496976, 1.0690449676496976]
    # price = predict(new_car, w, b)
    # print(price)


if __name__ == "__main__":
    main()
