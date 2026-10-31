
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

# --------------------------------------------------
# 2. Load trained models
# --------------------------------------------------

@st.cache_resource
def load_models():
    classifier = joblib.load(
        MODEL_DIR / "wine_quality_classifier_tuned.joblib"
    )

    regressor = joblib.load(
        MODEL_DIR / "wine_quality_regressor_tuned.joblib"
    )

    return classifier, regressor


try:
    classifier, regressor = load_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()


# --------------------------------------------------
# 3. Application title
# --------------------------------------------------

st.title("🍷 Wine Quality Prediction System")

st.write(
    "Predict wine quality using Machine Learning. "
    "This application provides both classification "
    "and regression predictions."
)

st.divider()

# --------------------------------------------------
# 4. Input form
# --------------------------------------------------

st.subheader("Enter Wine Characteristics")

with st.form("wine_prediction_form"):

    wine_type = st.selectbox(
        "Wine Type",
        ["red", "white"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input(
            "Fixed Acidity", min_value=0.0,
            value=7.0, step=0.1
        )

        volatile_acidity = st.number_input(
            "Volatile Acidity", min_value=0.0,
            value=0.3, step=0.01
        )

        citric_acid = st.number_input(
            "Citric Acid", min_value=0.0,
            value=0.3, step=0.01
        )

        residual_sugar = st.number_input(
            "Residual Sugar", min_value=0.0,
            value=5.0, step=0.1
        )

    with col2:
        chlorides = st.number_input(
            "Chlorides", min_value=0.0,
            value=0.05, step=0.001,
            format="%.3f"
        )

        free_sulfur_dioxide = st.number_input(
            "Free Sulfur Dioxide", min_value=0.0,
            value=30.0, step=1.0
        )

        total_sulfur_dioxide = st.number_input(
            "Total Sulfur Dioxide", min_value=0.0,
            value=100.0, step=1.0
        )

        density = st.number_input(
            "Density", min_value=0.0,
            value=0.995, step=0.0001,
            format="%.4f"
        )

    with col3:
        ph = st.number_input(
            "pH", min_value=0.0,
            value=3.2, step=0.01
        )

        sulphates = st.number_input(
            "Sulphates", min_value=0.0,
            value=0.5, step=0.01
        )

        alcohol = st.number_input(
            "Alcohol (%)", min_value=0.0,
            value=10.0, step=0.1
        )

    submitted = st.form_submit_button(
        "Predict Wine Quality",
        use_container_width=True
    )



# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "ph": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "wine_type": wine_type
    }])

    try:
        # Independent predictions from both models
        classification_prediction = int(
            classifier.predict(input_data)[0]
        )

        regression_prediction = float(
            regressor.predict(input_data)[0]
        )

        st.divider()
        st.header("Prediction Results")

        # ------------------------------------------
        # A. Regression result
        # ------------------------------------------

        st.subheader("1. Regression Prediction")

        st.metric(
            label="Predicted Wine Quality Score",
            value=f"{regression_prediction:.2f} / 10"
        )

        st.caption(
            "The regression model predicts a continuous "
            "wine quality score."
        )

        # ------------------------------------------
        # B. Classification result
        # ------------------------------------------

        st.subheader("2. Classification Prediction")

        if classification_prediction == 1:
            st.success(
                "Predicted Class: Good Quality (Quality >= 6)"
            )
        else:
            st.warning(
                "Predicted Class: Lower Quality (Quality < 6)"
            )

        st.caption(
            "The classification model independently predicts "
            "whether wine belongs to the lower-quality or "
            "good-quality category."
        )

        # ------------------------------------------
        # C. Classification probability
        # ------------------------------------------

        if hasattr(classifier, "predict_proba"):

            probabilities = classifier.predict_proba(
                input_data
            )[0]

            classes = list(classifier.classes_)

            if 1 in classes:
                positive_class_index = classes.index(1)

                positive_probability = (
                    probabilities[positive_class_index] * 100
                )

                st.metric(
                    "Probability of Good Quality",
                    f"{positive_probability:.2f}%"
                )

        # ------------------------------------------
        # D. Explanation
        # ------------------------------------------

        st.divider()

        st.info(
            "Why can the predictions differ? "
            "Regression and classification use separate "
            "trained models with different learning objectives. "
            "Therefore, their predictions may occasionally "
            "disagree. This is expected."
        )

        st.caption(
            "These are model predictions, not laboratory "
            "measurements or guaranteed wine-quality assessments."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
