import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title and description
st.title("Employee Safety Gear Detection Using Deep Learning (MobileNetV2)")
st.write("Upload an image and the model will predict whether the employee is wearing safety gear.")

# Load the model (without caching)
def load_safety_model():
    model = load_model('employee_safety_model.keras')
    return model

# Load the model
model = load_safety_model()

# Function to make prediction
def predict(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction using the loaded model
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "Wearing Safety Gear"
    else:
        return "Not Wearing Safety Gear"

# Upload an image
image_file = st.file_uploader("Upload an image of an employee", type=["jpg", "png", "jpeg"])

if image_file:
    # Display the image
    img = Image.open(image_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict if the person is wearing safety gear
    result = predict(image_file)
    st.write(f"Prediction: **{result}**")

