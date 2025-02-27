import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

# Model URL (Replace 'pyash14' with your GitHub username)
MODEL_URL = "https://github.com/pyash14/Mango-Disease-Detection/raw/main/model/MoblieNet_mango.h5"
MODEL_PATH = "MoblieNet_mango.h5"

# Download model if not available
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... Please wait")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)
st.success("Model loaded successfully!")

# Class labels
class_indices = {'Anthracnose': 0, 'Bacterial Canker': 1, 'Cutting Weevil': 2, 'Die Back': 3, 
                 'Gall Midge': 4, 'Healthy': 5, 'Powdery Mildew': 6, 'Sooty Mould': 7}
inv_class_indices = {v: k for k, v in class_indices.items()}

# Streamlit UI
st.title("Mango Leaf Disease Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Load image
    img = image.load_img(uploaded_file, target_size=(224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    # Predict
    pred = model.predict(img_array)
    predicted_class = np.argmax(pred)

    # Show results
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write(f"Predicted Label: {inv_class_indices[predicted_class]}")
    st.write(f"Confidence Score: {pred[0][predicted_class]:.4f}")
