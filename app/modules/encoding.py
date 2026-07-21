"""Encoding recommendations and application."""

from __future__ import annotations

from typing import Any

import pandas as pd
from category_encoders import BinaryEncoder, HashingEncoder, TargetEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from app.utils.helpers import get_column_types


def recommend_encoding(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    types = get_column_types(df)
    rows = []
    for col in types["categorical"]:
        if col == target_col:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 2:
            encoding = "label"
            reason = "Binary category — label encoding"
        elif n_unique <= 10:
            encoding = "onehot"
            reason = "Low cardinality — one-hot encoding"
        elif n_unique <= 50:
            encoding = "target" if target_col else "binary"
            reason = "Medium cardinality — target/binary encoding"
        else:
            encoding = "hashing"
            reason = "High cardinality — hashing encoding"
        rows.append(
            {
                "column": col,
                "unique_values": n_unique,
                "recommended_encoding": encoding,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def apply_encoding(
    df: pd.DataFrame,
    target_col: str,
    encodings: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    log: dict[str, Any] = {"encoders": {}, "columns_encoded": []}
    types = get_column_types(result)

    if encodings is None:
        rec = recommend_encoding(result, target_col)
        encodings = {
            row["column"]: row["recommended_encoding"] for _, row in rec.iterrows()
        }

    y = result[target_col]
    X = result.drop(columns=[target_col])

    for col, method in encodings.items():
        if col not in X.columns:
            continue
        if method == "label":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            log["encoders"][col] = "label"
        elif method == "onehot":
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = X.drop(columns=[col]).join(dummies)
            log["encoders"][col] = f"onehot ({len(dummies.columns)} cols)"
        elif method == "target":
            te = TargetEncoder()
            X[col] = te.fit_transform(X[[col]], y)
            log["encoders"][col] = "target"
        elif method == "binary":
            be = BinaryEncoder()
            encoded = be.fit_transform(X[[col]])
            X = X.drop(columns=[col]).join(encoded)
            log["encoders"][col] = f"binary ({len(encoded.columns)} cols)"
        elif method == "hashing":
            he = HashingEncoder(n_components=8)
            encoded = he.fit_transform(X[[col]])
            X = X.drop(columns=[col]).join(encoded)
            log["encoders"][col] = "hashing (8 components)"
        log["columns_encoded"].append(col)

    result = pd.concat([X, y], axis=1)
    return result, log
