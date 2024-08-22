import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

st.set_page_config(layout="wide")

# Initialize session state to store the current image
if "current_image" not in st.session_state:
    st.session_state.current_image = "green"

col1, col2 = st.columns([2, 1])

with col1:
    st.title("Heart Disease Prediction")

    # Sample data with standardized column names
    sample_data = {
        "age": 70,
        "sex": 1,
        "chest_pain_type": 4,
        "resting_blood_pressure": 130,
        "serum_cholestoral": 322,
        "fasting_blood_sugar": 0,
        "resting_electrocardiographic_results": 2,
        "max_heart_rate": 109,
        "exercise_induced_angina": 0,
        "oldpeak": 2.4,
        "st_segment": 2,
        "major_vessels": 3,
        "thal": 3,
    }

    st.write("Please enter the patient's information:")

    # Input fields (unchanged)
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=sample_data["age"],
        help="Patient's age in years",
    )
    sex = st.selectbox(
        "Sex",
        [0, 1],
        index=sample_data["sex"],
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="Patient's sex (0 = Female, 1 = Male)",
    )
    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4],
        index=sample_data["chest_pain_type"] - 1,
        help="Type of chest pain experienced: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic",
    )
    resting_blood_pressure = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        max_value=300,
        value=sample_data["resting_blood_pressure"],
        help="Resting blood pressure (in mm Hg)",
    )
    serum_cholestoral = st.number_input(
        "Serum Cholesterol",
        min_value=0,
        max_value=600,
        value=sample_data["serum_cholestoral"],
        help="Serum cholesterol in mg/dl",
    )
    fasting_blood_sugar = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        index=sample_data["fasting_blood_sugar"],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
    )
    resting_electrocardiographic_results = st.selectbox(
        "Resting ECG Results",
        [0, 1, 2],
        index=sample_data["resting_electrocardiographic_results"],
        help="Resting electrocardiographic results: 0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy",
    )
    max_heart_rate = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=0,
        max_value=300,
        value=sample_data["max_heart_rate"],
        help="Maximum heart rate achieved",
    )
    exercise_induced_angina = st.selectbox(
        "Exercise Induced Angina",
        [0, 1],
        index=sample_data["exercise_induced_angina"],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Exercise induced angina (1 = yes; 0 = no)",
    )
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise",
        min_value=0.0,
        max_value=10.0,
        value=sample_data["oldpeak"],
        step=0.1,
        help="ST depression induced by exercise relative to rest",
    )
    st_segment = st.selectbox(
        "Slope of the Peak Exercise ST Segment",
        [1, 2, 3],
        index=sample_data["st_segment"] - 1,
        help="The slope of the peak exercise ST segment: 1 = upsloping, 2 = flat, 3 = downsloping",
    )
    major_vessels = st.selectbox(
        "Number of Major Vessels Colored by Fluoroscopy",
        [0, 1, 2, 3],
        index=sample_data["major_vessels"],
        help="Number of major vessels (0-3) colored by fluoroscopy",
    )
    thal = st.selectbox(
        "Thal",
        [3, 6, 7],
        index=[3, 6, 7].index(sample_data["thal"]),
        help="Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect",
    )

with col2:
    st.write("")
    st.write("")
    green_image = Image.open("./streamlit/artifacts/green.png")
    red_image = Image.open("./streamlit/artifacts/red.png")

    # Create a container for vertical centering
    with st.container():
        # Add some vertical space
        st.write("")
        st.write("")

        # Create placeholders for image and text
        image_placeholder = st.empty()
        prediction_placeholder = st.empty()
        probability_placeholder = st.empty()

        # Display current image based on session state
        current_image = (
            green_image if st.session_state.current_image == "green" else red_image
        )
        image_placeholder.image(current_image, use_column_width=True)

if st.button("Predict"):
    input_data = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "serum_cholestoral": serum_cholestoral,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_electrocardiographic_results": resting_electrocardiographic_results,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_angina": exercise_induced_angina,
        "oldpeak": oldpeak,
        "st_segment": st_segment,
        "major_vessels": major_vessels,
        "thal": thal,
    }

    try:
        load_dotenv()

        # Get the API URL from the environment variable
        api_url = os.getenv("API_URL")

        # Build the complete URL
        url = f"{api_url}/predict"

        response = requests.post(url, json=input_data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                prediction = (
                    "Heart Disease" if result["prediction"] == 1 else "No Heart Disease"
                )
                color = "red" if result["prediction"] == 1 else "green"
                st.session_state.current_image = (
                    "red" if result["prediction"] == 1 else "green"
                )
                image = red_image if result["prediction"] == 1 else green_image

                image_placeholder.image(image, use_column_width=True)
                prediction_placeholder.markdown(
                    f"<h2 style='color: {color}; text-align: center;'>Prediction: {prediction}</h2>",
                    unsafe_allow_html=True,
                )

                if "probability" in result:
                    probability_placeholder.markdown(
                        f"<p style='text-align: center;'>Probability of Heart Disease: {result['probability']:.2f}</p>",
                        unsafe_allow_html=True,
                    )
            else:
                st.error(
                    "The API response doesn't contain a 'prediction' key. Please check the API implementation."
                )
        else:
            st.error(f"API request failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the request: {e}")
        st.write("Please make sure the FastAPI server is running and accessible.")
