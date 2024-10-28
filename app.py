import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import json

# Load the model
model = tf.keras.models.load_model("EuroSAT_model.h5")

# Load class labels
with open("label_map.json", "r") as f:
    class_labels = json.load(f)

# Reverse the class labels to get index to class name mapping
class_labels = {v: k for k, v in class_labels.items()}

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalizing
    return image

# Streamlit App
st.title("Satellite Image Classification")

st.write("Upload a satellite image, and the model will predict its class.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)
    class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[class_idx]
    
    # Show the predicted class
    st.write(f"Predicted Class: {predicted_class}")
