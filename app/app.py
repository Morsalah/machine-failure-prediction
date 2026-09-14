from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st


# Demo values only - update min/max ranges later if needed.

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide",
)


# =========================================================
# Paths and constants
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_final.joblib"

THRESHOLD = 0.09


# =========================================================
# Model loading
# =========================================================

@st.cache_resource
def load_model():
    """Load the trained machine learning pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# =========================================================
# Helper functions
# =========================================================

def clean_feature_name(name: str) -> str:
    """Make transformed feature names easier to read in the UI."""

    return (
        name
        .replace("numeric__", "")
        .replace("categorical__", "")
        .replace("Type_", "Machine Type: ")
    )


def create_shap_dataframe(
    model,
    input_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate SHAP values for a single prediction."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_input = preprocessor.transform(
        input_data
    )

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        transformed_input
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": shap_values[0],
        }
    )

    return shap_df


# =========================================================
# Load model safely
# =========================================================

try:
    model = load_model()

except Exception as error:
    st.error(
        "The machine learning model could not be loaded."
    )

    st.exception(error)

    st.stop()


# =========================================================
# Page header
# =========================================================

st.title("Machine Failure Prediction")

st.write(
    "Enter the current machine operating conditions "
    "to estimate failure risk."
)


# =========================================================
# Machine operating conditions
# =========================================================

st.subheader("Machine Operating Conditions")

col1, col2 = st.columns(2)


with col1:
    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1,
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=4000,
        value=1500,
        step=10,
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=500,
        value=100,
        step=1,
    )


with col2:
    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1,
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1,
    )

    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
    )


predict_button = st.button(
    "Predict Failure Risk",
    type="primary",
)


# =========================================================
# Prediction
# =========================================================

if predict_button:

    input_data = pd.DataFrame(
        {
            "Air temperature [K]": [
                air_temperature
            ],
            "Process temperature [K]": [
                process_temperature
            ],
            "Rotational speed [rpm]": [
                rotational_speed
            ],
            "Torque [Nm]": [
                torque
            ],
            "Tool wear [min]": [
                tool_wear
            ],
            "Type": [
                machine_type
            ],
        }
    )

    # Run prediction using the full pipeline.
    failure_probability = (
        model.predict_proba(
            input_data
        )[0, 1]
    )

    prediction = (
        failure_probability >= THRESHOLD
    )


    # =====================================================
    # Prediction Result
    # =====================================================

    st.divider()

    st.subheader(
        "Prediction Result"
    )

    result_col1, result_col2 = st.columns(
        [2, 1]
    )


    # -----------------------------------------------------
    # Failure probability gauge
    # -----------------------------------------------------

    with result_col1:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=(
                    failure_probability
                    * 100
                ),

                number={
                    "suffix": "%",
                    "valueformat": ".1f",
                },

                title={
                    "text": (
                        "Failure Probability"
                    )
                },

                gauge={
                    "axis": {
                        "range": [
                            0,
                            100,
                        ]
                    },

                    "bar": {
                        "color": (
                            "#F28E6B"
                            if prediction
                            else "#8EC5E8"
                        )
                    },

                    "steps": [
                        {
                            "range": [
                                0,
                                THRESHOLD * 100,
                            ],
                            "color": "#EAF4FA",
                        },
                        {
                            "range": [
                                THRESHOLD * 100,
                                100,
                            ],
                            "color": "#FCEAE4",
                        },
                    ],

                    "threshold": {
                        "line": {
                            "color": "#24364B",
                            "width": 4,
                        },

                        "thickness": 0.8,

                        "value": (
                            THRESHOLD * 100
                        ),
                    },
                },
            )
        )

        gauge.update_layout(
            height=300,

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20,
            ),
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )


    # -----------------------------------------------------
    # Threshold + decision
    # -----------------------------------------------------

    with result_col2:

        st.metric(
            label="Decision Threshold",

            value=(
                f"{THRESHOLD:.0%}"
            ),
        )

        if prediction:

            st.error(
                "High Failure Risk - "
                "Maintenance Recommended"
            )

        else:

            st.success(
                "Low Failure Risk - "
                "No Immediate Maintenance Alert"
            )


    st.caption(
        "The classification threshold was optimized "
        "on the validation set to prioritize failure "
        "detection in this highly imbalanced dataset."
    )


    # =====================================================
    # SHAP Explanation
    # =====================================================

    shap_df = create_shap_dataframe(
        model=model,
        input_data=input_data,
    )


    # -----------------------------------------------------
    # Clean feature names
    # -----------------------------------------------------

    shap_df["Feature"] = (
        shap_df["Feature"]
        .apply(
            clean_feature_name
        )
    )


    # -----------------------------------------------------
    # Combine one-hot encoded Machine Type features
    # -----------------------------------------------------

    shap_df["Feature"] = (
        shap_df["Feature"]
        .apply(
            lambda name: (
                "Machine Type"

                if name.startswith(
                    "Machine Type:"
                )

                else name
            )
        )
    )


    shap_df = (
        shap_df
        .groupby(
            "Feature",
            as_index=False,
        )["SHAP Value"]
        .sum()
    )


    # -----------------------------------------------------
    # Current user input values
    # -----------------------------------------------------

    feature_values = {

        "Air temperature [K]":
            f"{air_temperature:.1f} K",

        "Process temperature [K]":
            f"{process_temperature:.1f} K",

        "Rotational speed [rpm]":
            f"{rotational_speed} rpm",

        "Torque [Nm]":
            f"{torque:.1f} Nm",

        "Tool wear [min]":
            f"{tool_wear} min",

        "Machine Type":
            machine_type,
    }


    shap_df["Current Value"] = (
        shap_df["Feature"]
        .map(
            feature_values
        )
    )


    # -----------------------------------------------------
    # Direction of impact
    # -----------------------------------------------------

    shap_df["Effect"] = (
        shap_df["SHAP Value"]
        .apply(
            lambda value: (
                "Increases failure risk"

                if value > 0

                else "Decreases failure risk"
            )
        )
    )


    # Absolute impact is only used for sorting.
    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"].abs()
    )


    plot_data = (
        shap_df
        .sort_values(
            "Absolute Impact",
            ascending=True,
        )
    )


    # =====================================================
    # Interactive SHAP chart
    # =====================================================

    st.subheader(
        "Why this prediction?"
    )


    fig = px.bar(
        plot_data,

        x="SHAP Value",

        y="Feature",

        orientation="h",

        color="Effect",

        color_discrete_map={
            "Increases failure risk":
                "#F28E6B",

            "Decreases failure risk":
                "#8EC5E8",
        },

        custom_data=[
            "Current Value",
            "Effect",
        ],
    )


    # -----------------------------------------------------
    # Interactive hover information
    # -----------------------------------------------------

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"

            "Current Value: "
            "%{customdata[0]}<br>"

            "%{customdata[1]}<br>"

            "SHAP Impact: "
            "%{x:.3f}"

            "<extra></extra>"
        )
    )


    fig.update_layout(

        xaxis_title=(
            "Impact on Failure Risk"
        ),

        yaxis_title=None,

        legend_title=None,

        height=420,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )


    # Zero separates increasing and decreasing impact.
    fig.add_vline(
        x=0,
        line_width=1,
        line_color="gray",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.caption(
        "Features on the right increase the predicted failure risk, "
        "while features on the left decrease it. "
        "Hover over a bar for more details."
    )