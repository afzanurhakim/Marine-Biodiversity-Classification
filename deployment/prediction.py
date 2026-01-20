import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from pathlib import Path

def main():
    st.markdown(
        "Unggah gambar (JPEG/PNG) untuk diklasifikasikan ke dalam 5 kategori: "
        "**Dolphin, Fish, Lobster, Octopus, atau Sea Horse**."
    )
    st.markdown("---")

    # Load model (safe path)
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "marine_classification.keras"
    model = load_model(MODEL_PATH)

    uploaded_file = st.file_uploader(
        "Pilih Gambar Biota Laut...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")

        img = image.load_img(uploaded_file, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred_probs = model.predict(img_array)
        pred_class = np.argmax(pred_probs)

        class_names = ['Dolphin', 'Fish', 'Lobster', 'Octopus', 'Sea Horse']

        st.subheader("🔍 Prediction Result")
        st.write("### Jenis Biota Laut:", class_names[pred_class])
        st.write("### Confidence:", f"{float(np.max(pred_probs)):.4f}")

    else:
        st.info("Silakan upload gambar biota laut terlebih dahulu.")

if __name__ == "__main__":
    main()