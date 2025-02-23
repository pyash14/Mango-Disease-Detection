import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

MODEL_PATH = "MobileNet_mango.h5"
GDRIVE_FILE_ID = "1--qqwvwVwkhyHh1qnIdHT7KmckBpo6h7"  # Replace with your Google Drive file ID

# Check if model exists, if not, download it
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = [
    'Bacterial Canker', 'Anthracnose', 'Sooty Mold', 'Powdery Mildew',
    'Healthy', 'Dieback', 'Cutting Weevil', 'Gall Midge'
]

def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# Streamlit UI
st.title("Mango Disease Classification using MobileNet")
st.write("Upload an image of a mango leaf, and the model will predict its disease category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))

    st.subheader("Prediction:")
    st.write(f"**Predicted Disease:** {class_labels[predicted_class]}")
    st.write(f"**Confidence Score:** {confidence:.2f}")

    st.progress(int(confidence * 100))
