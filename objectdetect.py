import streamlit as st
import os
from model_handler import get_model, train_model, predict_image
from data_handler import save_uploaded_images_with_labels, preprocess_image
from PIL import Image

# Constants
UPLOAD_DIR = 'uploaded_images'
MODEL_PATH = 'object_detection_model.keras'
CLASS_NAMES_FILE = 'class_names.txt'
CONFIDENCE_THRESHOLD = 0.8  # Updated threshold to 80%

# Initialize session state
if 'classes' not in st.session_state:
    st.session_state['classes'] = []

# Function to load class names from a file
def load_class_names():
    if os.path.exists(CLASS_NAMES_FILE):
        with open(CLASS_NAMES_FILE, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []

# Function to add a new class to class_names.txt if it doesn't exist
def add_class_to_file(class_name):
    if not os.path.exists(CLASS_NAMES_FILE):
        with open(CLASS_NAMES_FILE, 'w') as f:
            f.write(f"{class_name}\n")
    else:
        with open(CLASS_NAMES_FILE, 'r') as f:
            existing_classes = [line.strip() for line in f.readlines()]
        if class_name not in existing_classes:
            with open(CLASS_NAMES_FILE, 'a') as f:
                f.write(f"{class_name}\n")

# Enhance UI for better user experience
st.set_page_config(page_title="Object Detection Trainer", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Add a sidebar for navigation
st.sidebar.title("Navigation")
options = ["Add Classes & Upload Images", "Train Model", "Test Model"]
choice = st.sidebar.radio("Go to", options)

if choice == "Add Classes & Upload Images":
    st.header('Add Classes and Upload Images')
    new_class_name = st.text_input("Enter new class name:", key="new_class")
    if st.button("Add Class"):
        if new_class_name and new_class_name not in st.session_state['classes']:
            st.session_state['classes'].append(new_class_name)
            add_class_to_file(new_class_name)  # Add to class_names.txt
            st.success(f"Class '{new_class_name}' added successfully!")
        elif new_class_name in st.session_state['classes']:
            st.warning(f"Class '{new_class_name}' already exists.")
        else:
            st.error("Please enter a valid class name.")

    for class_name in st.session_state['classes']:
        st.subheader(f"Upload Images for Class: {class_name}")
        uploaded_files = st.file_uploader(f"Choose images for class '{class_name}'", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"upload_{class_name}")
        if uploaded_files:
            valid_files = []
            for uploaded_file in uploaded_files:
                try:
                    Image.open(uploaded_file).verify()  # Verify if the file is a valid image
                    valid_files.append(uploaded_file)
                except Exception:
                    st.error(f"{uploaded_file.name} is not a valid image file.")

            if valid_files:
                with st.spinner(f"Saving images for class '{class_name}'..."):
                    save_uploaded_images_with_labels(valid_files, class_name, UPLOAD_DIR)
                st.success(f"Images for class '{class_name}' uploaded successfully!")

elif choice == "Train Model":
    st.header('Train the Model')
    if st.button('Train Model'):
        if len(os.listdir(UPLOAD_DIR)) == 0:
            st.error("No images found in the dataset. Please upload images before training.")
        else:
            with st.spinner("Training the model. This may take a while..."):
                model = get_model(MODEL_PATH, UPLOAD_DIR)
                train_model(model, UPLOAD_DIR, MODEL_PATH)
            st.success('Model trained successfully!')

elif choice == "Test Model":
    st.header('Test the Model')
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please train the model first.")
    else:
        test_file = st.file_uploader("Upload a single image to test", type=["jpg", "jpeg", "png"], key="test_with_output")
        if test_file is not None:
            test_image_path = os.path.join(UPLOAD_DIR, test_file.name)
            image = Image.open(test_file)
            image.save(test_image_path)
            with st.spinner("Processing the image and predicting..."):
                model = get_model(MODEL_PATH, UPLOAD_DIR)
                image_array = preprocess_image(test_image_path)
                predicted_class, confidence = predict_image(model, image_array)
            st.image(test_image_path, caption="Uploaded Image", use_container_width=True)

            class_names = load_class_names()
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class_name = "Unknown"
            elif predicted_class < len(class_names):
                predicted_class_name = class_names[predicted_class]
            else:
                predicted_class_name = f"Class {predicted_class}"  # Fallback

            st.write(f"Predicted Class: {predicted_class_name}")
            st.write(f"Confidence: {confidence:.2f}")