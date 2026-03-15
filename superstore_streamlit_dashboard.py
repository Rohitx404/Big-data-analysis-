
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Page title
st.title("📊 Superstore Profit Prediction Dashboard")
st.write("Enter transaction details to predict whether the company will get Profit or Loss.")

# -------------------------
# USER INPUT FORM
# -------------------------
with st.form("prediction_form"):

    sales = st.number_input(
        "Enter Sales Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    discount = st.slider(
        "Select Discount",
        min_value=0.0,
        max_value=1.0,
        value=0.1
    )

    quantity = st.number_input(
        "Enter Quantity",
        min_value=1,
        value=1,
        step=1
    )

    predict_btn = st.form_submit_button("Predict Profit")

# -------------------------
# MODEL PREDICTION
# -------------------------
if predict_btn:

    input_data = [[sales, discount, quantity]]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction > 0:
        st.success(f"Expected Profit: {prediction:.2f}")
        st.info("Suggestion: Transaction likely profitable")
    else:
        st.error(f"Expected Loss: {prediction:.2f}")
        st.warning("Suggestion: High discount or low sales may cause loss")

# -------------------------
# DATA VISUALIZATION
# -------------------------
st.header("📈 Sales vs Profit Analysis")

file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    ax.scatter(df["Sales"], df["Profit"])
    ax.set_xlabel("Sales")
    ax.set_ylabel("Profit")
    ax.set_title("Sales vs Profit Relationship")

    st.pyplot(fig)
