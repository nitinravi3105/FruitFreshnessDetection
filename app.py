import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("fruit_model.h5")

class_names = [
    'fresh apples',
    'fresh banana',
    'fresh bittergroud',
    'fresh capsicum',
    'fresh cucumber',
    'fresh okra',
    'fresh oranges',
    'fresh potato',
    'fresh tamto',

    'rotten apples',
    'rotten banana',
    'rotten bittergroud',
    'rotten capsicum',
    'rotten cucumber',
    'rotten okra',
    'rotten oranges',
    'rotten potato',
    'rotten tamto'
]

IMG_SIZE = 224

def preprocess(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title(" Fruit & Vegetable Freshness Detection")

option = st.radio("Choose Input Method:",
                  ["Upload Image", "Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)

        processed = preprocess(image)
        prediction = model.predict(processed)

        label = class_names[np.argmax(prediction)]

        st.success(f"Prediction: {label}")

elif option == "Webcam":
    camera_image = st.camera_input("Capture Image")

    if camera_image:
        image = Image.open(camera_image)
        st.image(image)

        processed = preprocess(image)
        prediction = model.predict(processed)

        label = class_names[np.argmax(prediction)]

        st.success(f"Prediction: {label}")
