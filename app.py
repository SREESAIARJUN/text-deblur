import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import cv2

# Set title
st.title("Text Image Deblurring with Deep Learning")

# Load the saved model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("text_deblur_model.keras")
    return model

model = load_model()

# Configuration
TARGET_SIZE = (256, 256)

def preprocess_image(img: Image.Image):
    """Convert PIL image to model-ready input (grayscale, [0,1], target size)."""
    img = img.convert("L")
    img = img.resize(TARGET_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def postprocess_image(pred_array):
    """Convert model output to displayable PIL image."""
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
    # Read and show input
    pil_blurred = Image.open(uploaded_file)
    st.image(pil_blurred, caption="Uploaded Blurred Image", use_column_width='auto')

    # Prepare and predict
    input_arr = preprocess_image(pil_blurred)
    pred = model.predict(input_arr)
    pil_deblurred = postprocess_image(pred)

    # Show output
    st.image(pil_deblurred, caption="Deblurred Output", use_column_width='auto')

    # Optional: Side-by-side comparison
    st.markdown("### Comparison")
    col1, col2 = st.columns(2)
    col1.image(pil_blurred.resize(TARGET_SIZE), caption="Blurred (resized)")
    col2.image(pil_deblurred, caption="Deblurred")

    # Optional: Compute PSNR/SSIM if user uploads a *sharp* ground truth
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

    # Allow to download result
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
