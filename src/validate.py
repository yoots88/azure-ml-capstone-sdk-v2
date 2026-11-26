"""Shared schema for feature preparation, training and online inference."""
from math import isfinite

SCHEMA_VERSION = "fda-source-v3"
CATEGORICAL = [
    "country_area", "state", "prior_project_area", "prior_product_type",
    "prior_additional_details",
]
NUMERIC = [
    "cum_inspections", "cum_non_compliant", "prior_non_compliant_rate",
    "prev_is_non_compliant", "days_since_last_insp", "years_since_last_insp",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "is_non_compliant"


def validate_split(frame):
    if list(frame.columns) != FEATURES + [TARGET]:
        raise ValueError(f"Unexpected feature schema: {list(frame.columns)}")
    for name in CATEGORICAL:
        if not frame[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{name} must contain nonempty strings.")
    for name in NUMERIC:
        if frame[name].dtype.kind not in "iuf" or not frame[name].map(isfinite).all():
            raise ValueError(f"{name} must contain finite numbers.")
    count, oai, rate, prev, days, years = (frame[x] for x in NUMERIC)
    if not ((count >= 0) & (oai >= 0) & (oai <= count)
            & rate.between(0, 1) & prev.isin([0, 1])
            & (days >= 0) & (years >= 0)).all():
        raise ValueError("Invalid inspection history.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Each split must contain both binary labels.")


def validate_payload(payload, numeric_columns=None):
    data = payload["input_data"]
    columns = FEATURES if numeric_columns is None else numeric_columns
    if data["columns"] != columns or not data["data"]:
        raise ValueError("Expected the ordered FDA feature columns and at least one row.")
    if len(data["index"]) != len(data["data"]):
        raise ValueError("The index and row counts differ.")
    for row in data["data"]:
        if len(row) != len(columns):
            raise ValueError("Unexpected feature count.")
        if numeric_columns is not None:
            if any(type(value) not in (int, float) or not isfinite(value) for value in row):
                raise ValueError("AutoML inputs must be finite numbers.")
            continue
        if any(not isinstance(x, str) or not x.strip() for x in row[:len(CATEGORICAL)]):
            raise ValueError("Location and prior categories must be nonempty strings.")
        values = row[len(CATEGORICAL):]
        if any(type(x) not in (int, float) or not isfinite(x) for x in values):
            raise ValueError("History features must be finite numbers.")
        count, oai, rate, previous, days, years = values
        if not (0 <= oai <= count and 0 <= rate <= 1 and previous in (0, 1)
                and days >= 0 and years >= 0):
            raise ValueError("Invalid inspection history.")
