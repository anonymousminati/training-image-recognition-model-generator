import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_num_classes(upload_dir):
    if isinstance(upload_dir, str):
        return len([name for name in os.listdir(upload_dir) if os.path.isdir(os.path.join(upload_dir, name))])
    else:
        raise TypeError("upload_dir must be a string representing the path to the directory.")

def update_model_for_new_classes(model, num_classes):
    # Extract the base model from the existing model
    if hasattr(model, 'layers') and len(model.layers) > 1:
        base_model = model.layers[0]  # Assuming the base model is the first layer
    else:
        raise ValueError("The provided model does not have a valid base model.")

    # Ensure the base model has an input
    if not hasattr(base_model, 'input') or base_model.input is None:
        raise ValueError("The base model does not have a valid input. Recreate the base model.")

    base_model.trainable = False

    # Create a new output layer with the updated number of classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create a new model with the updated output layer
    updated_model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the updated model
    updated_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return updated_model

def get_model(model_path, upload_dir):
    num_classes = get_num_classes(upload_dir)
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model.output_shape[-1] != num_classes:
            # Update the model for the new number of classes
            model = update_model_for_new_classes(model, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def train_model(model, upload_dir, model_path):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        upload_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        upload_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
    )
    model.fit(train_generator, validation_data=validation_generator, epochs=5)
    model.save(model_path)

def predict_image(model, image_array):
    predictions = model.predict(image_array)
    predicted_class = predictions.argmax(axis=1)[0]
    confidence = predictions.max()
    return predicted_class, confidence