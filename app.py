import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import cv2
import os
import requests
from pathlib import Path

# Set title
st.title("Text Image Deblurring with Deep Learning")

# Load the saved model with auto-download from GitHub Releases
@st.cache_resource
def load_model():
    model_path = Path("src/text_deblur_model.keras")
    if not model_path.exists():
        # Download from GitHub Releases
        st.info("Downloading model from GitHub...")
        url = "https://github.com/SREESAIARJUN/text-deblur/releases/download/1.0/text_deblur_model.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("Model downloaded successfully.")
    return tf.keras.models.load_model(str(model_path))

model = load_model()

# Configuration
TARGET_SIZE = (256, 256)

def preprocess_image(img: Image.Image):
    img = img.convert("L")
    img = img.resize(TARGET_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_image(pred_array):
    pred_img = np.clip(pred_array[0, :, :, 0], 0, 1) * 255
    pred_img = pred_img.astype(np.uint8)
    return Image.fromarray(pred_img, mode="L")

def calculate_psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(y_true, y_pred):
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(y_true, y_pred, data_range=1.0)
    except:
        return None

st.write("Upload a *blurred* text image (PNG/JPG). The model will attempt to restore it.")

uploaded_file = st.file_uploader("Choose a blurred image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    pil_blurred = Image.open(uploaded_file)
    st.image(pil_blurred, caption="Uploaded Blurred Image", use_container_width='auto')

    input_arr = preprocess_image(pil_blurred)
    pred = model.predict(input_arr)
    pil_deblurred = postprocess_image(pred)

    st.image(pil_deblurred, caption="Deblurred Output", use_container_width='auto')

    st.markdown("### Comparison")
    col1, col2 = st.columns(2)
    col1.image(pil_blurred.resize(TARGET_SIZE), caption="Blurred (resized)")
    col2.image(pil_deblurred, caption="Deblurred")

    with st.expander("Have a ground-truth sharp image? Upload to evaluate PSNR/SSIM:"):
        gt_file = st.file_uploader("Sharp groundtruth image", key="gt")
        if gt_file:
            pil_gt = Image.open(gt_file).convert("L").resize(TARGET_SIZE)
            gt_array = np.asarray(pil_gt, dtype=np.float32) / 255.0
            pred_array = np.asarray(pil_deblurred, dtype=np.float32) / 255.0
            psnr = calculate_psnr(gt_array, pred_array)
            ssim = calculate_ssim(gt_array, pred_array)
            st.write(f"**PSNR:** {psnr:.2f} dB")
            st.write(f"**SSIM:** {ssim:.4f}" if ssim is not None else "SSIM computation unavailable.")

    buf = pil_deblurred
    st.download_button(
        label="Download Deblurred Output",
        data=buf.tobytes(),
        file_name="deblurred.png",
        mime="image/png"
    )

st.markdown("""
---
*Powered by [Streamlit](https://streamlit.io/), Keras & your Attention U-Net Model*
""")
