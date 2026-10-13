"""ML Zoomcamp 2026 - Homework 3 (Machine Learning for Classification).

Reproducible solution for the lead-scoring dataset. Running this file prints
every answer required by the homework.

    pip install -r requirements.txt
    python solution.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/course_lead_scoring_2026.csv"
TARGET = "converted"
RANDOM_STATE = 42


def load_and_prepare(path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the CSV and fill missing values (cats -> 'NA', nums -> 0.0)."""
    df = pd.read_csv(path)
    features = df.drop(columns=[TARGET]).copy()

    categorical = features.select_dtypes(include=["object"]).columns.tolist()
    numerical = features.select_dtypes(exclude=["object"]).columns.tolist()

    features[categorical] = features[categorical].fillna("NA")
    features[numerical] = features[numerical].fillna(0.0)
    return features.assign(**{TARGET: df[TARGET]}), categorical, numerical


def train_model(train_x, train_y, val_x, c: float = 1.0):
    """One-hot encode and fit a logistic regression; return (model, val_matrix)."""
    dv = DictVectorizer(sparse=False)
    train_matrix = dv.fit_transform(train_x.to_dict(orient="records"))
    val_matrix = dv.transform(val_x.to_dict(orient="records"))

    model = LogisticRegression(
        solver="liblinear", C=c, max_iter=1000, random_state=RANDOM_STATE
    )
    model.fit(train_matrix, train_y)
    return model, val_matrix


def main() -> None:
    prepared, categorical, numerical = load_and_prepare(DATA_PATH)

    # Q1 - mode of `industry`
    industry_mode = prepared["industry"].mode().iloc[0]
    print(f"Q1. industry mode: {industry_mode}")

    # Q2 - strongest correlation among the suggested pairs
    corr = prepared[numerical].corr().abs()
    candidate_pairs = [
        ("interaction_count", "lead_score"),
        ("number_of_courses_viewed", "lead_score"),
        ("number_of_courses_viewed", "interaction_count"),
        ("annual_income", "interaction_count"),
    ]
    best_pair = max(candidate_pairs, key=lambda p: corr.loc[p[0], p[1]])
    print(
        f"Q2. biggest correlation: {best_pair[0]} and {best_pair[1]} "
        f"({corr.loc[best_pair[0], best_pair[1]]:.3f})"
    )

    # Split exactly as the homework requires.
    full_train, _test = train_test_split(
        prepared, test_size=0.2, random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        full_train, test_size=0.25, random_state=RANDOM_STATE
    )
    train_y = train.pop(TARGET)
    val_y = val.pop(TARGET)

    # Q3 - mutual information (training set only, rounded to 2 decimals)
    mi = {
        col: round(mutual_info_score(train_y, train[col]), 2)
        for col in categorical
    }
    print(f"Q3. mutual information: {mi}")
    print(f"Q3. biggest MI: {max(mi, key=mi.get)}")

    # Q4 - baseline logistic regression accuracy
    model, val_matrix = train_model(train, train_y, val)
    original_accuracy = accuracy_score(val_y, model.predict(val_matrix))
    print(f"Q4. accuracy: {original_accuracy:.2f}")

    # Q5 - feature elimination (smallest absolute accuracy difference)
    differences = {}
    for col in ["lead_source", "number_of_courses_viewed", "interaction_count"]:
        reduced_model, reduced_matrix = train_model(
            train.drop(columns=[col]), train_y, val.drop(columns=[col])
        )
        reduced_accuracy = accuracy_score(
            val_y, reduced_model.predict(reduced_matrix)
        )
        differences[col] = abs(original_accuracy - reduced_accuracy)
    print(f"Q5. elimination differences: {differences}")
    print(f"Q5. least useful feature: {min(differences, key=differences.get)}")

    # Q6 - regularized logistic regression (best C, smallest if tied)
    scores = {}
    for c in [0.000001, 0.00001, 0.0001, 0.001]:
        c_model, c_matrix = train_model(train, train_y, val, c=c)
        scores[c] = accuracy_score(val_y, c_model.predict(c_matrix))
    print(f"Q6. C scores: { {c: round(s, 3) for c, s in scores.items()} }")
    best_c = max(scores, key=lambda c: (round(scores[c], 3), -c))
    print(f"Q6. best C: {best_c:g}")


if __name__ == "__main__":
    main()
