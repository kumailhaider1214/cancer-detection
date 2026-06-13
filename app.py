import streamlit as st
from ultralytics import YOLO
import os
import shutil
from PIL import Image
import uuid

# Title
st.set_page_config(page_title="MRI Brain Tumor Detector", layout="centered")
st.title("🧠 MRI Brain Tumor Detection")

# Load YOLO model once
@st.cache_resource
def load_model():
    try:
        return YOLO("best2.pt")
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded image
    temp_filename = f"{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())

    # Prepare columns for side-by-side display
    col1, col2 = st.columns(2)
    col1.image(temp_filename, caption="Original Image", use_container_width=True)

    detected_image_path = None
    # Run YOLOv8 on image
    if model is not None:
        with st.spinner("Detecting Brain Tumor..."):
            try:
                results = model(temp_filename, save=True)
            except Exception as e:
                st.error(f"Error running YOLO prediction: {e}")
                results = None

            if results is not None and isinstance(results, list) and len(results) > 0:
                # Get the path of the detected image from the first results object using save_dir and filename
                try:
                    detected_image_path = os.path.join(str(results[0].save_dir), os.path.basename(temp_filename))
                    if not os.path.exists(detected_image_path):
                        st.error(f"Detected image not found at {detected_image_path}")
                        detected_image_path = None
                except Exception as e:
                    st.error(f"Could not get detected image path: {e}")

    # Show detected image in col2 if available
    if detected_image_path:
        col2.image(detected_image_path, caption="Detected Image", use_container_width=True)

    # Clean up if needed
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
