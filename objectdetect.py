import streamlit as st
import os
import json
from model_handler import get_model, train_model, predict_image
from data_handler import save_uploaded_images_with_labels, preprocess_image
from PIL import Image
import river
from river.naive_bayes import GaussianNB
import numpy as np

# Constants
UPLOAD_DIR = 'uploaded_images'
MODEL_PATH = 'object_detection_model.keras'
CLASS_NAMES_FILE = 'class_names.json'
CONFIDENCE_THRESHOLD = 0.8  # Updated threshold to 80%

# Initialize session state
if 'classes' not in st.session_state:
    st.session_state['classes'] = []

# Function to load class names from a JSON file
def load_class_names():
    if os.path.exists(CLASS_NAMES_FILE):
        with open(CLASS_NAMES_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save class names to a JSON file
def save_class_names(class_names):
    with open(CLASS_NAMES_FILE, 'w') as f:
        json.dump(class_names, f, indent=4)

# Function to add a new class to class_names.json if it doesn't exist
def add_class_to_file(class_name):
    class_names = load_class_names()
    if class_name not in class_names:
        class_names[class_name] = len(class_names)
        save_class_names(class_names)

# Automatically update class_names.json when new classes are detected
def update_class_names(upload_dir):
    class_names = load_class_names()
    current_classes = [name for name in os.listdir(upload_dir) if os.path.isdir(os.path.join(upload_dir, name))]

    new_classes = [cls for cls in current_classes if cls not in class_names]
    if new_classes:
        for cls in new_classes:
            class_names[cls] = len(class_names)
        save_class_names(class_names)
        st.success(f"New classes added: {', '.join(new_classes)}")

# Initialize a River model for incremental learning
pipeline = GaussianNB()

# Function to extract features from an image
def extract_image_features(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to a fixed size (e.g., 28x28)
    return np.array(image).flatten()  # Flatten the image into a 1D array

# Ensure the pipeline object is not overwritten
def train_incremental_model(pipeline, upload_dir, epochs=1):
    if pipeline is None:
        raise ValueError("The pipeline is not initialized. Please ensure it is properly set up before training.")

    class_names = load_class_names()
    for epoch in range(epochs):
        st.write(f"Epoch {epoch + 1}/{epochs}")  # Display the current epoch
        for class_name, class_index in class_names.items():
            class_dir = os.path.join(upload_dir, class_name)
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    if os.path.isfile(file_path):
                        try:
                            features = extract_image_features(file_path)  # Extract features from the image
                            features_dict = {f'pixel_{i}': float(value) for i, value in enumerate(features)}  # Ensure values are floats
                            pipeline.learn_one(features_dict, class_index)  # Call learn_one without reassigning pipeline
                        except Exception as e:
                            st.error(f"Error processing file {file_name}: {e}")
    return pipeline

# Function to predict using the River model
def predict_incremental_model(pipeline, file_path):
    with open(file_path, 'r') as f:
        text = f.read()  # Assuming text data for simplicity
        return pipeline.predict_one(text)

# Enhance UI for better user experience
st.set_page_config(page_title="Object Detection Trainer", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
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
            add_class_to_file(new_class_name)  # Add to class_names.json
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
    update_class_names(UPLOAD_DIR)  # Ensure class_names.json is up-to-date
    epochs = st.number_input("Enter the number of epochs:", min_value=1, value=1, step=1)
    if st.button('Train Model'):
        if len(os.listdir(UPLOAD_DIR)) == 0:
            st.error("No images found in the dataset. Please upload images before training.")
        else:
            with st.spinner("Training the model incrementally. This may take a while..."):
                pipeline = train_incremental_model(pipeline, UPLOAD_DIR, epochs=epochs)
                # Save the trained model
                with open(MODEL_PATH, 'wb') as f:
                    import pickle
                    pickle.dump(pipeline, f)
            st.success('Model trained incrementally and saved successfully!')

elif choice == "Test Model":
    st.header('Test the Model')
    test_file = st.file_uploader("Upload a single image to test", type=["jpg", "jpeg", "png"], key="test_with_output")
    if test_file is not None:
        # Save the uploaded file to a temporary location
        test_image_path = os.path.join(UPLOAD_DIR, test_file.name)
        with open(test_image_path, 'wb') as f:
            f.write(test_file.getbuffer())

        # Load the saved model
        try:
            with open(MODEL_PATH, 'rb') as f:
                import pickle
                pipeline = pickle.load(f)

            # Process the saved file
            features = extract_image_features(test_image_path)  # Extract features from the image
            features_dict = {f'pixel_{i}': float(value) for i, value in enumerate(features)}
            predicted_class = pipeline.predict_one(features_dict)

            # Map the predicted class index to the class name
            class_names = load_class_names()
            predicted_class_name = [name for name, index in class_names.items() if index == predicted_class]
            predicted_class_name = predicted_class_name[0] if predicted_class_name else "Unknown"

            st.write(f"Predicted Class: {predicted_class_name}")
        except Exception as e:
            st.error(f"Error loading the model or processing the test image: {e}")