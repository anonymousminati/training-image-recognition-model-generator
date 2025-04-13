# Object Detection Trainer

This project is a Streamlit-based application for training and testing object detection models. Users can upload images, label them with class names, train a model using transfer learning, and test the model with new images.

## Updated Features
- **Batch Testing**: Upload multiple images for testing, provide feedback using checkboxes, and view a bar chart showing counts of true predictions, false predictions, and unknowns.
- **Progress Bars**: Visual progress bars for each epoch during training.
- **Confidence Threshold**: Predictions below the confidence threshold (default: 80%) are classified as "Unknown."

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
4. Use the "Batch Test Model" section to test the model with multiple images and view overall accuracy.

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
  - The default value is 1, but you can increase it as needed.
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

### 4. Batch Test the Model
- Navigate to the "Batch Test Model" section from the sidebar.
- Upload multiple images for batch testing.
  - Each image will be processed, and the model will predict the class and confidence level.
- Provide feedback for each prediction using checkboxes to mark whether the prediction is correct.
- View the overall accuracy based on user feedback.
- A bar chart will display the counts of true predictions, false predictions, and unknowns.

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

## How Model Training Works

### Overview
The model training process in this application is designed to be incremental, meaning it updates the model with new data without retraining from scratch. This is achieved using the `river` package, which is specifically built for online machine learning. The training process is modular and efficient, allowing for real-time updates and handling of new classes dynamically.

### Packages and Libraries Used
1. **`river`**:
   - **Why Used**: The `river` package is ideal for online learning, where data arrives in a stream, and the model needs to be updated incrementally. It supports various machine learning algorithms and is lightweight and memory-efficient.
   - **Key Components**:
     - `GaussianNB`: A Naive Bayes classifier for numerical data, used for its simplicity and effectiveness in handling high-dimensional data like image features.

2. **`Pillow`**:
   - **Why Used**: To preprocess images by resizing and converting them to grayscale, making them suitable for feature extraction.

3. **`Streamlit`**:
   - **Why Used**: To create an interactive user interface for uploading images, training the model, and testing predictions.

4. **`NumPy`**:
   - **Why Used**: To handle numerical operations, such as flattening image arrays into feature vectors.

### Functions Used
1. **`train_incremental_model`**:
   - **Purpose**: Handles the incremental training of the model.
   - **Steps**:
     1. Extracts features from images using the `extract_image_features` function.
     2. Converts the features into a dictionary format required by the `river` package.
     3. Updates the model incrementally using the `learn_one` method of `GaussianNB`.
     4. Displays progress bars for each epoch to provide real-time feedback to the user.

2. **`extract_image_features`**:
   - **Purpose**: Preprocesses images by resizing them to a fixed size (28x28), converting them to grayscale, and flattening them into a 1D array of pixel values.
   - **Why**: Simplifies the image data into a format suitable for numerical classifiers like `GaussianNB`.

3. **`save_model` and `load_model`**:
   - **Purpose**: Save the trained model to disk and load it for future use.
   - **Why**: Ensures the model's state is persistent across sessions, allowing for incremental updates without losing prior knowledge.

4. **`update_class_names`**:
   - **Purpose**: Dynamically updates the list of class names based on the uploaded images.
   - **Why**: Ensures the model can handle new classes without requiring manual updates.

### Training Process
![alt text](<object detect algorithm diagram.png>)
1. **Data Preparation**:
   - Users upload images for each class via the "Add Classes & Upload Images" section.
   - Images are saved in the `uploaded_images/` directory, organized by class.

2. **Feature Extraction**:
   - Each image is preprocessed using the `extract_image_features` function.
   - The resulting feature vectors are converted into a dictionary format.

3. **Incremental Training**:
   - The `train_incremental_model` function iterates through the dataset for the specified number of epochs.
   - For each image, the `learn_one` method of `GaussianNB` updates the model with the new data.
   - Progress bars provide real-time feedback on the training process.

4. **Model Saving**:
   - After training, the model is saved to the `MODEL_PATH` file using Python's `pickle` module.
   - This ensures the model's state is preserved for future use.

### Why Incremental Learning?
- **Efficiency**: Incremental learning avoids the need to retrain the model from scratch, saving time and computational resources.
- **Scalability**: The model can handle new data and classes dynamically, making it suitable for real-world applications where data evolves over time.
- **Real-Time Updates**: Allows for immediate updates to the model as new data becomes available.

### Steps in Detail
1. **Upload Images**:
   - Users upload images for each class.
   - The application organizes the images into class-specific folders.

2. **Train the Model**:
   - Users specify the number of epochs for training.
   - The application preprocesses the images, extracts features, and updates the model incrementally.
   - Progress bars provide visual feedback during training.

3. **Save the Model**:
   - The trained model is saved to disk for future use.

4. **Test the Model**:
   - Users can test the model with single or multiple images.
   - The application predicts the class and confidence level for each image.
   - Batch testing allows users to provide feedback and view accuracy metrics.

By following this process, the application ensures an efficient and user-friendly workflow for training and testing object detection models.

### How Image Features Are Extracted

Image feature extraction is a crucial step in the model training process. The features are extracted using the following steps:

1. **Image Preprocessing**:
   - Each image is loaded using the `Pillow` library.
   - The image is converted to grayscale to reduce complexity and focus on intensity values.
   - The image is resized to a fixed size of 28x28 pixels to standardize the input dimensions.

2. **Flattening the Image**:
   - The resized grayscale image is converted into a 1D array of pixel intensity values.
   - This array represents the features of the image, where each value corresponds to the intensity of a pixel.

3. **Why This Approach?**:
   - **Simplicity**: Flattening the image into a 1D array simplifies the data structure, making it compatible with numerical classifiers like `GaussianNB`.
   - **Efficiency**: Reducing the image size and converting it to grayscale minimizes computational overhead while retaining essential information.
   - **Compatibility**: The extracted features are converted into a dictionary format, which is required by the `river` package for incremental learning.

4. **Code Implementation**:
   - The feature extraction is implemented in the `extract_image_features` function:
     ```python
     def extract_image_features(image_path):
         image = Image.open(image_path).convert('L')  # Convert to grayscale
         image = image.resize((28, 28))  # Resize to a fixed size (e.g., 28x28)
         return np.array(image).flatten()  # Flatten the image into a 1D array
     ```

By following this approach, the application ensures that the image data is preprocessed and formatted in a way that is both efficient and effective for training the model incrementally.