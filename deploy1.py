import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open(r"C:\Users\NIKHIL\Downloads\FedEx Project\Fedex.pkl", 'rb'))

# Streamlit App
st.title("FedEx Delivery Status Prediction")
st.write("Enter shipment details to predict the delivery status.")

# Input Fields
year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)
day_of_month = st.number_input("Day of Month", min_value=1, max_value=31, step=1)
day_of_week = st.number_input("Day of Week", min_value=1, max_value=7, step=1)
actual_shipment_time = st.number_input("Actual Shipment Time (HHMM)", min_value=0, max_value=2359, step=1)
planned_shipment_time = st.number_input("Planned Shipment Time (HHMM)", min_value=0, max_value=2359, step=1)
planned_delivery_time = st.number_input("Planned Delivery Time (HHMM)", min_value=0, max_value=2359, step=1)
carrier_name = st.text_input("Carrier Name")
carrier_num = st.text_input("Carrier Number")
planned_time_of_travel = st.number_input("Planned Time of Travel (minutes)", min_value=0, step=1)
shipment_delay = st.number_input("Shipment Delay (minutes)", step=1)
source = st.text_input("Source Location")
destination = st.text_input("Destination Location")
distance = st.number_input("Distance (miles)", min_value=0, step=1)

def get_user_input():
    data = {
        "Year": year,
        "Month": month,
        "DayofMonth": day_of_month,
        "DayOfWeek": day_of_week,
        "Actual_Shipment_Time": actual_shipment_time,
        "Planned_Shipment_Time": planned_shipment_time,
        "Planned_Delivery_Time": planned_delivery_time,
        "Carrier_Name": carrier_name,
        "Carrier_Num": carrier_num,
        "Planned_TimeofTravel": planned_time_of_travel,
        "Shipment_Delay": shipment_delay,
        "Source": source,
        "Destination": destination,
        "Distance": distance
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# One-Hot Encoding using `pd.get_dummies()`
input_df = pd.get_dummies(input_df)

# Ensure feature order matches the model
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict when button is clicked
if st.button("Predict"):
    prediction = model.predict(input_df)[0]  # Get single value

    if prediction == 1:
        st.success("✅ Delivered")
    else:
        st.error("❌ Not Delivered")


