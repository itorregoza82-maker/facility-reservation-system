
import pandas as pd
import plotly.express as px


def facility_summary(df):
    """Summarize reservation records by facility type."""
    if "Facility_Type" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("Facility_Type", dropna=False)
        .size()
        .reset_index(name="Total Reservations")
    )

    if "Reservation_Outcome" in df.columns:
        outcomes = pd.crosstab(
            df["Facility_Type"].fillna("Missing"),
            df["Reservation_Outcome"].fillna("Missing")
        ).reset_index()

        summary["Facility_Type"] = summary["Facility_Type"].fillna("Missing")
        summary = summary.merge(
            outcomes,
            on="Facility_Type",
            how="left"
        )

    return summary


def facility_utilization_chart(df):
    """Display the number of reservation records per facility type."""
    if "Facility_Type" not in df.columns:
        return None

    counts = (
        df["Facility_Type"]
        .fillna("Missing")
        .value_counts()
        .rename_axis("Facility Type")
        .reset_index(name="Reservation Records")
    )

    return px.bar(
        counts,
        x="Facility Type",
        y="Reservation Records",
        title="Reservation Records by Facility Type",
        hover_data=["Reservation Records"]
    )


def outcome_by_facility_chart(df):
    """Compare observed reservation outcomes across facility types."""
    required = {"Facility_Type", "Reservation_Outcome"}

    if not required.issubset(df.columns):
        return None

    counts = (
        df.assign(
            Facility_Type=df["Facility_Type"].fillna("Missing"),
            Reservation_Outcome=df["Reservation_Outcome"].fillna("Missing")
        )
        .groupby(["Facility_Type", "Reservation_Outcome"])
        .size()
        .reset_index(name="Count")
    )

    return px.bar(
        counts,
        x="Facility_Type",
        y="Count",
        color="Reservation_Outcome",
        barmode="group",
        title="Observed Outcomes by Facility Type"
    )


def review_flag(predictions):
    """Flag predictions for administrator review, not automatic rejection."""
    results = predictions.copy()

    if "Predicted_Outcome" in results.columns:
        results["Administrator_Review"] = (
            results["Predicted_Outcome"] == "Not Completed"
        ).map({
            True: "Review recommended",
            False: "No outcome-based flag"
        })

    return results
