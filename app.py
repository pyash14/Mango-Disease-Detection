import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set the correct path to your model file
MODEL_PATH = r"C:/Users/yash0/Downloads/MoblieNet_mango.h5"

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Model file not found at '{MODEL_PATH}'. Please check the path and restart the app.")
    st.stop()

# Load the trained MobileNet model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Update with your actual class names)
class_labels = [
    'Bacterial Canker', 'Anthracnose', 'Sooty Mold', 'Powdery Mildew',
    'Healthy', 'Dieback', 'Cutting Weevil', 'Gall Midge'
]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for MobileNet
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    return image

# Streamlit UI
st.title("Mango Disease Classification using MobileNet")
st.write("Upload an image of a mango leaf, and the model will predict its disease category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image (Updated to use use_container_width=True)
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and make a prediction
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))

    # Display prediction result
    st.subheader("Prediction:")
    st.write(f"**Predicted Disease:** {class_labels[predicted_class]}")
    st.write(f"**Confidence Score:** {confidence:.2f}")

    # Add a progress bar for confidence level
    st.progress(int(confidence * 100))

