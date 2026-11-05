import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]

df = pd.read_csv("data/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"],
    random_state=params["random_state"], stratify=y
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

with mlflow.start_run(run_name="KNN_Iris"):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1_score": f1_score(y_test, predictions, average="weighted"),
    }

    mlflow.log_param("model", "KNN")
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_param("test_size", params["test_size"])
    mlflow.log_param("random_state", params["random_state"])
    mlflow.log_metrics(metrics)
    print("Métriques KNN :", metrics)