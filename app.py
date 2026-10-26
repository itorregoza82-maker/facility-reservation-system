
import streamlit as st
import pandas as pd
import plotly.express as px

from modules.data_processing import (
    load_csv,
    validate_dataset,
    dataset_summary,
    MODEL_FEATURES,
    TARGET_COLUMN,
)
from modules.exploratory_analysis import (
    descriptive_statistics,
    outcome_distribution,
    facility_chart,
    outcome_chart,
    time_category_chart,
)
from modules.prediction import predict_reservations
from modules.dashboard import (
    facility_summary,
    facility_utilization_chart,
    outcome_by_facility_chart,
    review_flag,
)

st.set_page_config(
    page_title="Facility Reservation System",
    page_icon="🏢",
    layout="wide",
)

st.title("Facility Reservation System")
st.caption("Reservation outcome prediction and facility monitoring")

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Data Analysis",
        "Prediction",
        "Dashboard",
        "System Design",
    ],
)

def get_dataset():
    return st.session_state.dataset

if page == "Home":
    st.header("Home")
    st.write(
        "This system supports facility reservation monitoring by allowing "
        "users to upload reservation data, explore patterns, and review "
        "predicted reservation outcomes."
    )
    st.info(
        "Predictions are decision-support information only. "
        "An administrator must review cases; the system does not "
        "automatically approve or reject reservations."
    )
    st.subheader("Getting Started")
    st.write("1. Open **Dataset** and upload a CSV file.")
    st.write("2. Explore records and charts in **Data Analysis**.")
    st.write("3. Use **Prediction** to estimate reservation outcomes.")
    st.write("4. Review facility summaries in **Dashboard**.")

elif page == "Dataset":
    st.header("Dataset Upload and Validation")
    uploaded_file = st.file_uploader("Upload a reservation CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
            is_valid, errors = validate_dataset(df, require_target=False)

            if not is_valid:
                st.error("The dataset cannot be used yet.")
                for error in errors:
                    st.write(f"- {error}")
            else:
                st.session_state.dataset = df
                st.session_state.predictions = None
                st.success("Dataset uploaded and required model columns validated.")

                summary = dataset_summary(df)
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", summary["rows"])
                col2.metric("Columns", summary["columns"])
                col3.metric("Duplicate Rows", summary["duplicate_rows"])

                if summary["duplicate_reservation_ids"] is not None:
                    st.write(
                        "Duplicate Reservation_ID values:",
                        summary["duplicate_reservation_ids"],
                    )

                st.subheader("Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True)

                st.subheader("Missing Values by Column")
                st.dataframe(
                    summary["missing_by_column"].rename("Missing Values"),
                    use_container_width=True,
                )

                st.subheader("Descriptive Statistics")
                st.dataframe(
                    descriptive_statistics(df),
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Unable to process the uploaded file: {exc}")

    else:
        st.write("Upload your CSV to inspect and validate the dataset.")

elif page == "Data Analysis":
    st.header("Exploratory Data Analysis")
    df = get_dataset()

    if df is None:
        st.warning("Please upload a dataset first using the Dataset page.")
    else:
        st.subheader("Descriptive Statistics")
        st.dataframe(descriptive_statistics(df), use_container_width=True)

        charts = [
            ("Reservations by Facility Type", facility_chart(df)),
            ("Observed Reservation Outcomes", outcome_chart(df)),
            ("Reservations by Time Category", time_category_chart(df)),
        ]

        for title, chart in charts:
            if chart is not None:
                st.subheader(title)
                st.plotly_chart(chart, use_container_width=True)

        if TARGET_COLUMN in df.columns:
            st.subheader("Reservation Outcome Distribution")
            st.dataframe(outcome_distribution(df), use_container_width=True)
        else:
            st.info(
                "This dataset has no Reservation_Outcome column. "
                "Outcome distribution is unavailable, but valid feature "
                "columns can still be used for prediction."
            )

elif page == "Prediction":
    st.header("Reservation Outcome Prediction")
    df = get_dataset()

    if df is None:
        st.warning("Please upload a dataset first using the Dataset page.")
    else:
        st.write("The model uses these six input features:")
        st.code(", ".join(MODEL_FEATURES))

        st.caption(
            "The uploaded dataset must contain all six features with valid "
            "numeric values for Reservation_Lead_Days and Participants."
        )

        if st.button("Predict Reservation Outcomes"):
            try:
                st.session_state.predictions = predict_reservations(df)
                st.success("Predictions generated.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        if st.session_state.predictions is not None:
            results = st.session_state.predictions.copy()
            st.dataframe(results, use_container_width=True)

            combined = df.copy()
            for column in results.columns:
                combined[column] = results[column]

            csv_data = combined.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Dataset with Predictions",
                data=csv_data,
                file_name="facility_reservation_predictions.csv",
                mime="text/csv",
            )

            st.info(
                "Predictions are estimates, not guaranteed outcomes. "
                "Review cases using the institution's reservation policy "
                "before making any decision."
            )

elif page == "Dashboard":
    st.header("Facility Monitoring Dashboard")
    df = get_dataset()

    if df is None:
        st.warning("Please upload a dataset first using the Dataset page.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Reservation Records", len(df))
        col2.metric("Facility Types", df["Facility_Type"].nunique())
        col3.metric(
            "Observed Outcomes Available",
            "Yes" if TARGET_COLUMN in df.columns else "No",
        )

        chart = facility_utilization_chart(df)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

        outcome_chart_result = outcome_by_facility_chart(df)
        if outcome_chart_result is not None:
            st.plotly_chart(outcome_chart_result, use_container_width=True)

        st.subheader("Facility Summary")
        st.dataframe(facility_summary(df), use_container_width=True)

        if st.session_state.predictions is not None:
            st.subheader("Administrator Review Queue")
            review_results = review_flag(st.session_state.predictions)
            flagged = review_results[
                review_results.get(
                    "Administrator_Review",
                    pd.Series(index=review_results.index, dtype="object"),
                ) == "Review recommended"
            ]
            st.write(
                "These cases are flagged for human review only. "
                "A prediction must not automatically deny a reservation."
            )
            st.dataframe(flagged, use_container_width=True)

elif page == "System Design":
    st.header("System Design")
    st.subheader("Functional Requirements")
    st.write("1. The system shall accept reservation datasets in CSV format.")
    st.write("2. The system shall validate required model input columns.")
    st.write("3. The system shall display dataset summaries and visualizations.")
    st.write("4. The system shall predict reservation outcomes using a saved model.")
    st.write("5. The system shall present facility summaries and cases for administrator review.")

    st.subheader("Nonfunctional Requirements")
    st.write("1. Usability: navigation and outputs should be clear to users.")
    st.write("2. Reliability: invalid datasets and prediction errors should be reported.")
    st.write("3. Maintainability: data processing, analysis, prediction, and dashboard code are separated into modules.")

    st.subheader("Decision Workflow")
    st.code(
        "Upload CSV -> Validate Columns and Values -> Explore Data -> "
        "Generate Predictions -> Administrator Reviews Flagged Cases"
    )

    st.subheader("Use Case Diagram")
    from pathlib import Path
    diagram_path = Path(__file__).resolve().parent / "assets" / "use_case_diagram.png"
    if diagram_path.exists():
        st.image(str(diagram_path), caption="Facility Reservation System Use Case Diagram")
    else:
        st.warning(
            "The use case diagram image has not been added to the assets folder yet."
        )

st.sidebar.divider()
st.sidebar.caption("Decision support only • Human review required")
