import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('house_model.pkl')  # Adjust the filename as necessary

# Set up the title and description
st.title("Housing Price Prediction App")
st.write("Enter the details of the house to predict its price.")

# Create input fields for user input
area = st.number_input("Area (in sq ft)", min_value=0.0, value=0.0)
bedrooms = st.selectbox("Number of Bedrooms", options=list(range(1, 7)), index=0)
bathrooms = st.selectbox("Number of Bathrooms", options=list(range(1, 5)), index=0)
stories = st.selectbox("Number of Stories", options=list(range(1, 5)), index=0)
mainroad = st.selectbox("Is it on Main Road?", options=["Yes", "No"])
guestroom = st.selectbox("Does it have a Guestroom?", options=["Yes", "No"])
basement = st.selectbox("Does it have a Basement?", options=["Yes", "No"])
hotwaterheating = st.selectbox("Does it have Hot Water Heating?", options=["Yes", "No"])
airconditioning = st.selectbox("Does it have Air Conditioning?", options=["Yes", "No"])
parking = st.selectbox("Number of Parking Spaces", options=list(range(1, 4)), index=0)
prefarea = st.selectbox("Is it in a Preferred Area?", options=["Yes", "No"])
furnishingstatus = st.selectbox("Furnishing Status", options=["Furnished", "Semi-Furnished", "Unfurnished"])

# Convert categorical inputs to numerical values
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

# One-hot encoding for furnishing status
furnishingstatus_furnished = 1 if furnishingstatus == "Furnished" else 0
furnishingstatus_semi_furnished = 1 if furnishingstatus == "Semi-Furnished" else 0
furnishingstatus_unfurnished = 1 if furnishingstatus == "Unfurnished" else 0

# Create a DataFrame to match the feature names used in training
input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad,
                             guestroom, basement, hotwaterheating, airconditioning,
                             parking, prefarea, 
                             furnishingstatus_furnished, 
                             furnishingstatus_semi_furnished, 
                             furnishingstatus_unfurnished]], 
                           columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
                                    'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                                    'parking', 'prefarea', 
                                    'furnishingstatus_furnished', 
                                    'furnishingstatus_semi-furnished', 
                                    'furnishingstatus_unfurnished'])

# Predict button
if st.button("Predict Price"):
    # Make the prediction
    price_prediction = model.predict(input_data)

    # Display the result
    st.success(f"The predicted price should be: ${price_prediction[0]:,.2f}")
