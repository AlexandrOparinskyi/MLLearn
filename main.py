dataset = {
    "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "age": [25, 40, 30, 50, 22, 35, 28, 45, 33, 38],
    "gender": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
    "monthly_fee": [50, 70, 60, 80, 45, 65, 55, 75, 60, 68],
    "tenure_months": [12, 24, 8, 36, 5, 18, 10, 30, 14, 20],
    "support_calls": [1, 3, 0, 5, 2, 1, 4, 3, 0, 2],
    "churned": [0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
}
y = dataset.pop("churned")
print(dataset)