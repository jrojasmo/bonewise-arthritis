import streamlit as st
import keras
from keras.applications import efficientnet
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PATH_TO_MODEL = SCRIPT_DIR.parent / "model" / "best_knee_model_efficientnet.keras"
print(PATH_TO_MODEL)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Knee Arthritis Detection",
    page_icon="🩻",
    layout="centered"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model_from_keras():
    """Loads the Keras model from file."""
    return keras.models.load_model(PATH_TO_MODEL)

model = load_model_from_keras()

# --- CLASS DEFINITIONS ---
class_names = [
    'Grade 0: Normal',
    'Grade 1: Doubtful',
    'Grade 2: Mild',
    'Grade 3: Moderate',
    'Grade 4: Severe'
]

# --- IMAGE PREPROCESSING FUNCTION ---
def preprocess_image(image_pil, img_size):
    """
    Applies the full preprocessing pipeline to a user-uploaded PIL image.
    """
    img_array = np.array(image_pil)
    # Load image in grayscale
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Resize with a high-quality interpolation method
    img_resized = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    # CLAHE for contrast enhancement (good for medical images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)
     # Convert to 3 channels for the pre-trained model
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    # Use efficientnet.preprocess_input and add a batch dimension
    img_processed = efficientnet.preprocess_input(np.expand_dims(img_rgb, axis=0))
    
    return img_processed

# --- USER INTERFACE ---
col1, col2 = st.columns((1, 1), gap="large")

with col1:
    st.title("🩻 Knee Arthritis Classifier")
    st.markdown(
    """
    **Upload a knee X-ray image**, and the model will predict its arthritis grade  
    based on the **Kellgren-Lawrence (KL) scale**:

    - 🟢 **Grade 0**: Normal  
    - 🟡 **Grade 1**: Doubtful  
    - 🟠 **Grade 2**: Mild  
    - 🔶 **Grade 3**: Moderate  
    - 🔴 **Grade 4**: Severe  
    """
    )

    uploaded_file = st.file_uploader(
        "Choose an X-ray knee image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info('Awaiting an X-ray image to be uploaded.')
        st.stop()

with col2: 
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray.', use_container_width=True)
    st.divider()

    if st.button('Classify Image', use_container_width=True):
        with st.spinner('Analyzing the image...', show_time=True):
            IMG_SIZE = 224 
            processed_image = preprocess_image(image, IMG_SIZE)
            
            # --- PREDICTION ---
            predictions = model.predict(processed_image)
            score = keras.ops.softmax(predictions[0])

            # --- DISPLAY RESULTS ---
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            st.success(f"**Diagnosis:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            st.write("Class Probabilities:")
            chart_data = {name: prob for name, prob in zip([name.split(':')[1].strip() for name in class_names], np.array(score))}
            st.bar_chart(chart_data, horizontal=True)
    