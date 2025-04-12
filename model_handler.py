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

def get_model(model_path, upload_dir):
    num_classes = get_num_classes(upload_dir)
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model.output_shape[-1] != num_classes:
            raise ValueError(f"Model output shape ({model.output_shape[-1]}) does not match the number of classes ({num_classes}). Please retrain the model.")
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