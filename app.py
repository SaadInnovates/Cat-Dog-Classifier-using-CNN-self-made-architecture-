import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_binary_classifier.h5")

model = load_model()

# Streamlit UI
st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((128, 128))  # same as training size
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Probability
    if prediction > 0.5:
        confidence = prediction * 100
        st.write(f"### Prediction: Dog ({confidence:.2f}% confidence)")
    else:
        confidence = (1 - prediction) * 100
        st.write(f"### Prediction: Cat ({confidence:.2f}% confidence)")
