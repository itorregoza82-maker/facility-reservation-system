
import pandas as pd
import plotly.express as px


def descriptive_statistics(df):
    """Return descriptive statistics for the uploaded dataset."""
    return df.describe(include="all").transpose()


def outcome_distribution(df, target_column="Reservation_Outcome"):
    """Count observed reservation outcomes when the target is available."""
    if target_column not in df.columns:
        return None

    return (
        df[target_column]
        .fillna("Missing")
        .value_counts()
        .rename_axis("Reservation Outcome")
        .reset_index(name="Count")
    )


def facility_chart(df):
    """Interactive reservation count by facility type."""
    if "Facility_Type" not in df.columns:
        return None

    counts = (
        df["Facility_Type"]
        .fillna("Missing")
        .value_counts()
        .rename_axis("Facility Type")
        .reset_index(name="Reservations")
    )

    return px.bar(
        counts,
        x="Facility Type",
        y="Reservations",
        title="Reservations by Facility Type",
        hover_data=["Reservations"],
    )


def outcome_chart(df, target_column="Reservation_Outcome"):
    """Interactive observed outcome distribution."""
    distribution = outcome_distribution(df, target_column)

    if distribution is None:
        return None

    return px.pie(
        distribution,
        names="Reservation Outcome",
        values="Count",
        title="Observed Reservation Outcomes",
        hole=0.35,
    )


def time_category_chart(df):
    """Interactive reservation count by time category."""
    if "Time_Category" not in df.columns:
        return None

    counts = (
        df["Time_Category"]
        .fillna("Missing")
        .value_counts()
        .rename_axis("Time Category")
        .reset_index(name="Reservations")
    )

    return px.bar(
        counts,
        x="Time Category",
        y="Reservations",
        title="Reservations by Time Category",
        hover_data=["Reservations"],
    )
