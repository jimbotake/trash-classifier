import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Lokasi file ini (streamlit_app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path ke model (naik satu folder, lalu ke folder model)
MODEL_PATH = os.path.join(BASE_DIR, '../model/trash_cnn_10_epoch.h5')

# Load model
model = load_model(MODEL_PATH)

# Nama kelas sesuai dataset
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("Trash Classifier")

uploaded_file = st.file_uploader("Upload an image of trash", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess gambar supaya sesuai input model (150x150)
    image = ImageOps.fit(image, (150, 150))
    image_array = np.asarray(image) / 255.0
    image_array = image_array.reshape(1, 150, 150, 3)

    # Prediksi kelas
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction)
    class_label = class_names[class_idx]
    confidence = prediction[0][class_idx]

    st.write(f"Predicted Class: **{class_label}**")
    st.write(f"Confidence: {confidence:.2%}")
