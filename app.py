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
st.title("Cat/Dog Classifier")
st.write("Upload an image and let the model classify it as a cat or dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((128, 128))  # resize same as training size
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"Prediction: Dog ")
    else:
        st.error(f"Prediction: Cat ")
