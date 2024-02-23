import os
import json
from PIL import Image

import numpy as np
import streamlit as st
from tensorflow.python.platform import _pywrap_tf2
from tensorflow.python import tf2 as _tf2
import tensorflow as tf


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/Trained Model/wasteClassification.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
classes = json.load(open(f"{working_dir}/classes.json"))


# Function to Load and Preprocess the Image using Pillow
# prompt: function to load and preprocess the images

def load_and_preprocess_image(image_path):
  # Load the image from the file path.
  image = tf.io.read_file(image_path)

  # Decode the image from a JPEG string to a 3D tensor.
  image = tf.image.decode_jpeg(image, channels=3)

  # Resize the image to the desired size.
  image = tf.image.resize(image, size=(imgSize, imgSize))

  # Normalize the pixel values between 0 and 1.
  image = image / 255.0

  return image



# prompt: assign the two classes to "recycle" and "organic"

classes = {
    0: "organic",
    1: "recycle",
}

def predict_image_class(image_path):
  # Preprocess the image.
  image = load_and_preprocess_image(image_path)

  # Add a batch dimension to the image.
  image = tf.expand_dims(image, axis=0)

  # Predict the class of the image.
  predictions = model.predict(image)

  # Get the class with the highest probability.
  predicted_class = tf.argmax(predictions, axis=1)

  # Return the class name.
  return classes[predicted_class.numpy()[0]]


# Streamlit App
st.title('Waste Sorting Mechanism')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')