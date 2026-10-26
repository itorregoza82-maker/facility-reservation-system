
import pandas as pd

MODEL_FEATURES = [
    "Facility_Type",
    "User_Type",
    "Time_Category",
    "Day_of_Week",
    "Reservation_Lead_Days",
    "Participants",
]

TARGET_COLUMN = "Reservation_Outcome"

LEAKAGE_COLUMNS = [
    "Usage_Status",
    "Actual_Attendance",
    "Attendance_Ratio",
    "Cancellation_Days_Before",
]


def load_csv(uploaded_file):
    """Load a CSV file and return a DataFrame."""
    try:
        return pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Unable to read this CSV file: {exc}") from exc


def validate_dataset(df, require_target=False):
    """Check whether the dataset contains the required columns."""
    if df is None or df.empty:
        return False, ["The dataset is empty."]

    if len(df.columns) == 0:
        return False, ["The dataset has no columns."]

    required = MODEL_FEATURES.copy()
    if require_target:
        required.append(TARGET_COLUMN)

    missing = [column for column in required if column not in df.columns]

    if missing:
        return False, [f"Missing required columns: {', '.join(missing)}"]

    errors = []

    for column in ["Reservation_Lead_Days", "Participants"]:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(values.isna().sum())

        if invalid_count:
            errors.append(
                f"{column} contains {invalid_count} missing or non-numeric value(s)."
            )

    return len(errors) == 0, errors


def dataset_summary(df):
    """Return useful data-quality summaries."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_by_column": df.isna().sum(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_reservation_ids": (
            int(df["Reservation_ID"].duplicated().sum())
            if "Reservation_ID" in df.columns
            else None
        ),
    }


def clean_model_features(df):
    """Prepare model input columns without using outcome leakage fields."""
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {', '.join(missing)}")

    result = df[MODEL_FEATURES].copy()

    for column in ["Reservation_Lead_Days", "Participants"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if result.isna().any().any():
        raise ValueError(
            "Model inputs contain missing or invalid values. "
            "Please correct them before prediction."
        )

    return result
