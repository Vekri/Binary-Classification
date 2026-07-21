"""Hyperparameter tuning with Optuna."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from app.utils.helpers import safe_encode_target, split_features_target

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _prepare_data(df: pd.DataFrame, target_col: str, max_rows: int = 1500):
    X, y = split_features_target(df, target_col)
    y_enc, _ = safe_encode_target(y)
    X_proc = pd.get_dummies(X, drop_first=True).fillna(0)
    if len(X_proc) > max_rows:
        idx = X_proc.sample(n=max_rows, random_state=42).index
        X_proc, y_enc = X_proc.loc[idx], y_enc.loc[idx]
    return X_proc, y_enc


def tune_random_forest(X, y, n_trials: int = 10) -> dict[str, Any]:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 120),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "random_state": 42,
            "n_jobs": 1,
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X, y, cv=2, scoring="roc_auc", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_score": round(study.best_value, 4)}


def tune_gradient_boosting(X, y, n_trials: int = 10) -> dict[str, Any]:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "random_state": 42,
        }
        model = GradientBoostingClassifier(**params)
        return cross_val_score(model, X, y, cv=2, scoring="roc_auc", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_score": round(study.best_value, 4)}


def tune_logistic_regression(X, y, n_trials: int = 8) -> dict[str, Any]:
    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 0.01, 5, log=True),
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": 42,
        }
        model = LogisticRegression(**params)
        return cross_val_score(model, X, y, cv=2, scoring="roc_auc", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_score": round(study.best_value, 4)}


TUNERS = {
    "Random Forest": tune_random_forest,
    "Gradient Boosting": tune_gradient_boosting,
    "Logistic Regression": tune_logistic_regression,
}


def run_hyperparameter_tuning(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    n_trials: int = 10,
) -> dict[str, Any]:
    X, y = _prepare_data(df, target_col)
    tuner = TUNERS.get(model_name)
    if tuner is None:
        return {"error": f"No tuner available for {model_name}"}
    result = tuner(X, y, n_trials=n_trials)
    result["model_name"] = model_name
    result["n_trials"] = n_trials
    return result
