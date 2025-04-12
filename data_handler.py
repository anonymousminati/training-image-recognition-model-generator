import os
from PIL import Image
import numpy as np

def save_uploaded_images_with_labels(uploaded_files, class_name, upload_dir):
    class_dir = os.path.join(upload_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_path = os.path.join(class_dir, uploaded_file.name)
        image.save(image_path)

def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array