
import streamlit as st
import pandas as pd
import joblib
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("Tourism Package Purchase Prediction")
st.write("Enter customer details to predict tourism package purchase.")

# Load the trained model committed to the repository
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_model.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Collect customer inputs
st.subheader("Customer Information")

age = st.number_input("Age", min_value=18, max_value=100, value=35)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

citytier = st.selectbox("City Tier", [1, 2, 3])

durationofpitch = st.number_input(
    "Duration of Pitch", min_value=0, value=15
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

numberofpersonvisiting = st.number_input(
    "Number of Persons Visiting", min_value=1, value=2
)

numberoffollowups = st.number_input(
    "Number of Followups", min_value=0, value=3
)

productpitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Unmarried"]
)

numberoftrips = st.number_input(
    "Number of Trips", min_value=0, value=3
)

passport = st.selectbox("Passport", [0, 1])

pitchsatisfaction = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

owncar = st.selectbox("Own Car", [0, 1])

numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting", min_value=0, value=0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthlyincome = st.number_input(
    "Monthly Income", min_value=0, value=25000
)

# Prediction button
if st.button("Predict Tourism Package Purchase"):

    # Store user inputs in a dictionary
    input_data = {
        "Age": age,
        "TypeofContact": typeofcontact,
        "CityTier": citytier,
        "DurationOfPitch": durationofpitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": numberofpersonvisiting,
        "NumberOfFollowups": numberoffollowups,
        "ProductPitched": productpitched,
        "PreferredPropertyStar": preferredpropertystar,
        "MaritalStatus": maritalstatus,
        "NumberOfTrips": numberoftrips,
        "Passport": passport,
        "PitchSatisfactionScore": pitchsatisfaction,
        "OwnCar": owncar,
        "NumberOfChildrenVisiting": numberofchildrenvisiting,
        "Designation": designation,
        "MonthlyIncome": monthlyincome
    }

    # Convert inputs into a DataFrame
    input_df = pd.DataFrame([input_data])

    st.subheader("Customer Input Data")
    st.dataframe(input_df)

    # Generate prediction
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer is likely to purchase the tourism package.")
    else:
        st.warning("Customer is unlikely to purchase the tourism package.")

    # Display prediction probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_labels = list(model.classes_)

        if 1 in class_labels:
            probability = probabilities[class_labels.index(1)]
            st.metric(
                "Purchase Probability",
                f"{probability * 100:.2f}%"
            )
