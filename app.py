import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗"
)

st.title("🚗 Vehicle Image Classifier")
st.write("Image upload karein aur model vehicle predict karega.")

# Class names
class_names = [
    'airplane',
    'ambulance',
    'bicycle',
    'boat',
    'bus',
    'car',
    'fire_truck',
    'helicopter',
    'hovercraft',
    'jet_ski',
    'kayak',
    'motorcycle',
    'rickshaw',
    'scooter',
    'segway',
    'skateboard',
    'tractor',
    'truck',
    'unicycle',
    'van'
]

# Load model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("car_cnn_keras.keras")

model = load_my_model()

# Upload image
uploaded_file = st.file_uploader(
    "Vehicle ki image upload karein",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.image(img, caption="Uploaded Image", width=400)

    # Predict button
    if st.button("🔍 Predict"):

        # Resize
        img = img.resize((128, 128))

        # Convert to array
        image_array = np.array(img)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Prediction
        prediction = model.predict(image_array)

        # Get class
        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]

        # Confidence
        confidence = prediction[0][predicted_index] * 100

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")