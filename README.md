# Object Detection Trainer

This project is a Streamlit-based application for training and testing object detection models. Users can upload images, label them with class names, train a model using transfer learning, and test the model with new images.

## Features
- Upload images with class labels.
- Train a model using transfer learning.
- Test the model with new images.
- Automatically handles new classes and updates the model.
- Displays predictions with confidence levels.
- Classifies low-confidence predictions as "Unknown."

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
- `CONFIDENCE_THRESHOLD`: Confidence level below which predictions are classified as "Unknown."

## Usage
1. Navigate to the "Add Classes & Upload Images" section to upload images.
2. Go to the "Train Model" section to train the model.
3. Use the "Test Model" section to test the model with new images.

## Error Handling
- Invalid image files are skipped during upload.
- Training is not allowed if the dataset is empty.
- Testing is not allowed if the model is not trained.

## Future Improvements
- Add support for more advanced models.
- Implement real-time object detection.
- Enhance UI with more visualizations.