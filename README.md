# Object Detection Trainer

This project is a Streamlit-based application for training and testing object detection models. Users can upload images, label them with class names, train a model using transfer learning, and test the model with new images.

## Features
- Upload images with class labels.
- Train a model using transfer learning.
- Test the model with new images.
- Automatically handles new classes and updates the model.
- Displays predictions with confidence levels.
- Classifies low-confidence predictions as "Unknown."
- Progress bars for each epoch during training.

## Setup Instructions
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run objectdetect.py
   ```

## Configuration
- `UPLOAD_DIR`: Directory where uploaded images are stored.
- `MODEL_PATH`: Path to save/load the trained model.
- `CLASS_NAMES_FILE`: Path to the JSON file storing class names.
- `CONFIDENCE_THRESHOLD`: Confidence level below which predictions are classified as "Unknown."

## Usage
1. Navigate to the "Add Classes & Upload Images" section to upload images.
2. Go to the "Train Model" section to train the model. Specify the number of epochs and view progress bars for each epoch.
3. Use the "Test Model" section to test the model with new images. The model will classify the image and display the confidence level.

## Detailed Steps for Using the Application

### 1. Add Classes & Upload Images
- Navigate to the "Add Classes & Upload Images" section from the sidebar.
- Enter a new class name in the input field and click the "Add Class" button.
  - If the class already exists, a warning will be displayed.
  - If the class is successfully added, a success message will appear.
- For each class, upload images by selecting files using the file uploader.
  - Ensure the images are in `.jpg`, `.jpeg`, or `.png` format.
  - Uploaded images will be saved in the corresponding class folder under the `uploaded_images/` directory.
  - A success message will confirm the upload.

### 2. Train the Model
- Navigate to the "Train Model" section from the sidebar.
- Ensure that images are uploaded for all desired classes.
- Specify the number of epochs for training using the input field.
  - The default value is 30, but you can increase it as needed.
- Click the "Train Model" button to start training.
  - A progress bar will display the progress of each epoch.
  - The model will be trained incrementally, and its state will be saved after training.
  - A success message will confirm that the model has been trained and saved.

### 3. Test the Model
- Navigate to the "Test Model" section from the sidebar.
- Upload a single image to test the model.
  - The image will be temporarily saved and processed.
- The model will classify the image and display:
  - The predicted class name.
  - The confidence level of the prediction.
- If the confidence level is below the threshold (default: 80%), the result will be displayed as "Unknown."

### Notes
- Ensure that the `uploaded_images/` directory is structured with subfolders for each class.
- The model is saved after every training session to ensure persistence.
- Low-confidence predictions are classified as "Unknown" to handle uncertain cases.

## Error Handling
- Invalid image files are skipped during upload.
- Training is not allowed if the dataset is empty.
- Testing is not allowed if the model is not trained.
- Low-confidence predictions are classified as "Unknown."

## Future Improvements
- Add support for more advanced models.
- Implement real-time object detection.
- Enhance UI with more visualizations and interactivity.

## Folder Structure
- `uploaded_images/`: Contains subfolders for each class with their respective images.
- `class_names.json`: Stores class names and their corresponding indices.
- `objectdetect.py`: Main Streamlit application.
- `model_handler.py`: Handles model creation, training, and prediction.
- `data_handler.py`: Handles data-related operations like saving and preprocessing images.
- `README.md`: Documentation for the project.

## Notes
- Ensure that the `uploaded_images/` directory is structured with subfolders for each class.
- The model is saved after every training session to ensure persistence.