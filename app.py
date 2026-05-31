import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_cat_dog.h5")

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

st.title("Cat vs Dog Classification Using CNN")

tab1, tab2 = st.tabs(["Upload Image", "Model Information"])

with tab1:

    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose a Cat or Dog image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)
        img = image.resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        if st.button('Predict Image'):
            prediction = model.predict(img)[0][0]
            if prediction > 0.5:
                result = "🐶 Dog"
            else:
                result = "🐱 Cat"
            st.subheader("Prediction Result")
            st.success(result)
            st.write(f"Confidence Score: {float(prediction):.4f}")

with tab2:
    st.subheader("CNN Architecture")
    st.markdown("""
    - Conv2D (32 filters)
    - MaxPooling2D
    - Conv2D (32 filters)
    - MaxPooling2D
    - Conv2D (32 filters)
    - MaxPooling2D
    - Flatten
    - Dense (128 neurons)
    - Dense (1 neuron, Sigmoid)
    """)
    st.subheader("Input Shape")
    st.write("(128, 128, 3)")
    st.subheader("Output")
    st.write("Binary Classification: Cat or Dog")