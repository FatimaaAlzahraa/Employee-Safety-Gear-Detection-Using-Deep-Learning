import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title of the app
st.title("Employee Safety Gear Detection Using Deep Learning (MobileNetV2)")
st.write("Upload an image and the model will predict whether the employee is wearing safety gear.")

# Use st.cache_resource instead of st.cache to cache the model
@st.cache  # Use this only if you cannot update Streamlit
def load_safety_model():
    model = load_model(r'C:\Users\zahra\Downloads\projectdeep\employee_safety_model.keras')  # Replace with the correct model path
    return model

# model = load_safety_model()

# Function to preprocess and predict
def predict(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return "Not Wearing Safety Gear" if prediction[0][0] < 0.5 else " Wearing Safety Gear"

# Upload an image
image_file = st.file_uploader("Upload an image of an employee", type=["jpg", "png", "jpeg"])

if image_file:
    img = Image.open(image_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    result = predict(image_file)
    st.write(f"Prediction: **{result}**")
