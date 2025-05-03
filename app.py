import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
# import cv2 # Not strictly needed for the core logic shown, maybe for advanced metrics
import os
import requests
from pathlib import Path

# --- Configuration ---
TARGET_SIZE = (256, 256)
SAMPLE_IMG_DIR = "." # Directory where sample images are stored
SAMPLE_IMG_WIDTH = 150    # Display width for sample images in Streamlit

# --- Model Loading ---
@st.cache_resource
def load_model():
    model_path = Path("src/text_deblur_model.keras")
    if not model_path.exists():
        st.info("Downloading model from GitHub...")
        url = "https://github.com/SREESAIARJUN/text-deblur/releases/download/1.0/text_deblur_model.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with requests.get(url, stream=True, timeout=30) as r: # Added timeout
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                prog_bar = st.progress(0)
                bytes_downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size > 0:
                            prog_bar.progress(min(1.0, bytes_downloaded / total_size))
                    prog_bar.progress(1.0) # Ensure it completes
            st.success("Model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.stop() # Stop execution if model download fails
        except Exception as e:
            st.error(f"An error occurred during model download/saving: {e}")
            st.stop()

    # --- Model Loading Try-Except ---
    try:
        model = tf.keras.models.load_model(str(model_path))
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Keras model from {model_path}: {e}")
        st.error("Please ensure the downloaded file is a valid Keras model.")
        # Optionally remove the potentially corrupted file
        if model_path.exists():
             try:
                 os.remove(model_path)
                 st.info(f"Removed potentially corrupted model file: {model_path}")
             except OSError as oe:
                 st.warning(f"Could not remove corrupted model file: {oe}")
        st.stop() # Stop execution if model loading fails

# --- Helper Functions ---
def preprocess_image(img: Image.Image):
    img = img.convert("L") # Convert to grayscale
    img = img.resize(TARGET_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

def postprocess_image(pred_array):
    # Ensure prediction is squeezed to 3D (H, W, C) if it has a batch dim
    if pred_array.ndim == 4:
        pred_array = pred_array[0]
    # Ensure prediction is squeezed to 2D (H, W) if it has a channel dim of 1
    if pred_array.ndim == 3 and pred_array.shape[-1] == 1:
        pred_array = pred_array[:, :, 0]

    pred_img = np.clip(pred_array, 0, 1) * 255
    pred_img = pred_img.astype(np.uint8)
    return Image.fromarray(pred_img, mode="L")

def calculate_psnr(y_true, y_pred):
    # Ensure arrays are float32 and in range [0, 1]
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100.0 # Or float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(y_true, y_pred):
    try:
        # Ensure skimage is installed: pip install scikit-image
        from skimage.metrics import structural_similarity as ssim
        # Ensure arrays are float32 and in range [0, 1]
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        # data_range is the difference between max and min possible pixel values
        return ssim(y_true, y_pred, data_range=1.0, channel_axis=None) # Use channel_axis=None for grayscale
    except ImportError:
        st.warning("scikit-image not installed. Cannot calculate SSIM. Install with: pip install scikit-image")
        return None
    except Exception as e:
        st.error(f"Error calculating SSIM: {e}")
        return None

# --- Streamlit App UI ---

st.set_page_config(layout="wide") # Use wider layout

st.title("Text Image Deblurring with Deep Learning")
st.markdown("This app uses a deep learning model (Attention U-Net) to attempt to remove blur from text images.")

# --- Load Model ---
# Wrapped model loading in try-except block inside the function
model = load_model()

# --- Display Sample Images Section ---
st.markdown("---")
st.subheader("Examples")
st.write("Here are some sample blurred images and their corresponding sharp (ground truth) versions:")

sample_pairs = [
    ("blurred_0_000000.png", "sharp_0_000000.png"),
    ("blurred_0_000005.png", "sharp_0_000005.png"),
    ("blurred_0_000010.png", "sharp_0_000010.png"),
]

# Create columns for sample display layout
cols = st.columns(len(sample_pairs) * 2) # Two columns per pair (blurred, sharp)

col_index = 0
for blurred_fname, sharp_fname in sample_pairs:
    blurred_path = os.path.join(SAMPLE_IMG_DIR, blurred_fname)
    sharp_path = os.path.join(SAMPLE_IMG_DIR, sharp_fname)

    # Display Blurred Sample
    with cols[col_index]:
        st.markdown(f"**Blurred**")
        if os.path.exists(blurred_path):
            try:
                img_blurred = Image.open(blurred_path)
                st.image(img_blurred, caption=f"{blurred_fname}", width=SAMPLE_IMG_WIDTH)
            except Exception as e:
                st.warning(f"Cannot load {blurred_fname}: {e}")
        else:
            st.caption(f"{blurred_fname}\n(not found)")
    col_index += 1

    # Display Sharp Sample
    with cols[col_index]:
        st.markdown(f"**Sharp (GT)**")
        if os.path.exists(sharp_path):
            try:
                img_sharp = Image.open(sharp_path)
                st.image(img_sharp, caption=f"{sharp_fname}", width=SAMPLE_IMG_WIDTH)
            except Exception as e:
                st.warning(f"Cannot load {sharp_fname}: {e}")
        else:
            st.caption(f"{sharp_fname}\n(not found)")
    col_index += 1


# --- Main Application Logic ---
st.markdown("---")
st.subheader("Try it yourself!")
st.write("Upload a *blurred* text image (PNG/JPG/JPEG). The model will attempt to restore it.")

uploaded_file = st.file_uploader("Choose a blurred image...", type=["png", "jpg", "jpeg"])

if uploaded_file and model: # Check if model loaded successfully
    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.markdown("#### Uploaded Blurred Image")
        try:
            pil_blurred = Image.open(uploaded_file)
            # Display original uploaded image before resizing for processing
            st.image(pil_blurred, caption="Original Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error opening uploaded image: {e}")
            pil_blurred = None # Ensure variable is None if loading fails

    if pil_blurred: # Proceed only if image loaded successfully
        # Process and Predict
        input_arr = preprocess_image(pil_blurred)
        pred_arr = model.predict(input_arr)
        pil_deblurred = postprocess_image(pred_arr)

        with col_up2:
            st.markdown("#### Deblurred Output")
            st.image(pil_deblurred, caption="Deblurred Result", use_container_width=True)

            # Offer download for the deblurred image
            # Convert PIL image to bytes for download
            from io import BytesIO
            buf = BytesIO()
            pil_deblurred.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Deblurred Image (PNG)",
                data=byte_im,
                file_name=f"deblurred_{uploaded_file.name}",
                mime="image/png"
            )

        # --- Optional Ground Truth Comparison ---
        st.markdown("---")
        with st.expander("Optional: Evaluate with Ground Truth"):
            st.write("If you have the corresponding sharp (ground truth) image, upload it here to calculate PSNR and SSIM metrics.")
            gt_file = st.file_uploader("Upload Sharp Ground Truth Image", type=["png", "jpg", "jpeg"], key="gt_eval")

            if gt_file:
                try:
                    pil_gt = Image.open(gt_file).convert("L").resize(TARGET_SIZE)
                    gt_array = np.asarray(pil_gt, dtype=np.float32) / 255.0

                    # Ensure the predicted array matches GT dimensions for comparison
                    # Use the postprocessed PIL image, converted back to array
                    pred_array_eval = np.asarray(pil_deblurred, dtype=np.float32) / 255.0

                    # Ensure arrays have the same shape before metric calculation
                    if gt_array.shape == pred_array_eval.shape:
                        psnr = calculate_psnr(gt_array, pred_array_eval)
                        ssim = calculate_ssim(gt_array, pred_array_eval)

                        st.metric(label="PSNR (Peak Signal-to-Noise Ratio)", value=f"{psnr:.2f} dB")
                        if ssim is not None:
                           st.metric(label="SSIM (Structural Similarity Index)", value=f"{ssim:.4f}")
                        else:
                           st.write("SSIM calculation failed or is unavailable.")

                        # Display GT side-by-side with result for visual comparison
                        st.markdown("##### Ground Truth vs. Deblurred")
                        col_comp1, col_comp2 = st.columns(2)
                        col_comp1.image(pil_gt, caption="Ground Truth (Resized)", use_container_width=True)
                        col_comp2.image(pil_deblurred, caption="Deblurred Result", use_container_width=True)

                    else:
                        st.error(f"Shape mismatch between Ground Truth ({gt_array.shape}) and Prediction ({pred_array_eval.shape}). Cannot calculate metrics.")

                except Exception as e:
                    st.error(f"Error processing ground truth image or calculating metrics: {e}")


st.markdown("""
---
*Model based on Attention U-Net architecture. Powered by [Streamlit](https://streamlit.io/) and [TensorFlow/Keras](https://www.tensorflow.org/)*
""")
