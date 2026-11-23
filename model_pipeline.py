import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def prepare_data(filepath: str):
    """Charge et prétraite le dataset des patients diabétiques."""
    df = pd.read_csv(filepath, na_values=["?"], keep_default_na=False)
    df.replace("", np.nan, inplace=True)

    cols_to_drop = [
        "encounter_id",
        "weight",
        "payer_code",
        "medical_specialty",
        "examide",
        "citoglipton",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df["target"] = (df["readmitted"] == "<30").astype(int)

    groups = df["patient_nbr"]
    X = df.drop(columns=["readmitted", "target", "patient_nbr"])
    y = df["target"]

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Entraîne un modèle Random Forest équilibré."""
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle avec les métriques adaptées aux classes déséquilibrées."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
    }
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")
    return metrics


def save_model(model, filepath: str = "diabetes_model.joblib"):
    """Sauvegarde le modèle entraîné sur le disque avec joblib."""
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé sous : {filepath}")


def load_model(filepath: str = "diabetes_model.joblib"):
    """Charge un modèle sauvegardé depuis le disque."""
    return joblib.load(filepath)
