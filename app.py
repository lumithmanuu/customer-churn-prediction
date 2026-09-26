import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")


model = load_model()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn "
    "using a trained machine learning model."
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("About the Project")

    st.write(
        """
        This application uses a Logistic Regression model trained
        on the Telco Customer Churn dataset.
        """
    )

    st.write("**Model:** Logistic Regression")
    st.write("**ROC-AUC:** ~83.5%")
    st.write("**Accuracy:** ~80%")

    st.divider()

    st.caption(
        "Machine Learning portfolio project by Lumith Manujaya"
    )


# --------------------------------------------------
# Input form
# --------------------------------------------------

with st.form("customer_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen_text = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    with col2:
        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col3:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )


    st.divider()

    st.subheader("🌐 Services")

    col1, col2, col3 = st.columns(3)

    with col1:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    st.divider()

    st.subheader("💳 Billing Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col2:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col3:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )


    st.write("")

    submit = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submit:

    senior_citizen = (
        1 if senior_citizen_text == "Yes" else 0
    )

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = model.predict(customer)[0]

    churn_probability = (
        model.predict_proba(customer)[0][1]
    )

    probability_percent = (
        churn_probability * 100
    )


    st.divider()

    st.subheader("📈 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.error(
                "⚠️ Customer is predicted to CHURN"
            )
        else:
            st.success(
                "✅ Customer is predicted to NOT CHURN"
            )


    with result_col2:

        st.metric(
            label="Churn Probability",
            value=f"{probability_percent:.2f}%"
        )


    st.progress(churn_probability)


    if churn_probability >= 0.70:

        st.warning(
            "High churn risk"
        )

    elif churn_probability >= 0.40:

        st.info(
            "Moderate churn risk"
        )

    else:

        st.success(
            "Low churn risk"
        )


    with st.expander(
        "View Customer Data Used for Prediction"
    ):

        st.dataframe(
            customer,
            use_container_width=True
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Python • Scikit-learn • Streamlit"
)


