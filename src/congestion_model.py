"""Train and evaluate a classical ML classifier for congestion prediction.

Input features come from data/processed/features.csv (built by src/build_dataset.py).
We compare Random Forest vs Logistic Regression and keep whichever performs better
on the held-out test set -- not asserted, measured.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "vehicle_count", "occupancy_ratio",
    "car_ratio", "motorcycle_ratio", "bus_ratio", "truck_ratio",
]
TARGET_COLUMN = "congestion_label"


def load_dataset(csv_path: str = "data/processed/features.csv"):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def train_and_compare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # Random Forest: no scaling needed, handles non-linear feature interactions
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["RandomForest"] = (rf, accuracy_score(y_test, rf_preds), rf_preds)

    # Logistic Regression: needs scaled features to converge/perform fairly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    results["LogisticRegression"] = (lr, accuracy_score(y_test, lr_preds), lr_preds)

    for name, (_, acc, _) in results.items():
        print(f"{name}: test accuracy = {acc:.3f}")

    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_preds = results[best_name]
    print(f"\nBest model: {best_name} (accuracy = {best_acc:.3f})")
    print("\nClassification report:\n", classification_report(y_test, best_preds))
    print("Confusion matrix (rows=actual, cols=predicted):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    return best_name, best_model, scaler if best_name == "LogisticRegression" else None


if __name__ == "__main__":
    X, y = load_dataset()
    print(f"Loaded {len(X)} samples, class distribution:\n{y.value_counts()}\n")

    best_name, best_model, scaler = train_and_compare(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": best_model, "scaler": scaler, "model_name": best_name}, "models/congestion_clf.pkl")
    print(f"\nSaved best model ({best_name}) to models/congestion_clf.pkl")
