import streamlit as st
import pickle
import numpy as np
from PIL import Image

# --------------------------------------------------
# Load the trained model (EXACT filename)
# --------------------------------------------------
MODEL_PATH = "model_comparison_results (1).pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --------------------------------------------------
# GTSRB class labels (43 classes)
# --------------------------------------------------
CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End of speed limit 80",
    "Speed limit 100", "Speed limit 120", "No passing",
    "No passing for vehicles over 3.5 t",
    "Right-of-way at the next intersection", "Priority road",
    "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5 t prohibited", "No entry",
    "General caution", "Dangerous curve left",
    "Dangerous curve right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 t"
]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="GTSRB Traffic Sign Classifier", layout="centered")

st.title("🚦 German Traffic Sign Recognition")
st.write("Upload a traffic sign image to predict its class.")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing (must match training)
    img = image.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, -1)  # flatten

    # Prediction
    prediction = model.predict(img_array)
    class_id = int(prediction[0])

    st.success(f"✅ Predicted Sign: **{CLASS_NAMES[class_id]}**")
