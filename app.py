import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Initialize the pre-trained ResNet50 model
def initialize_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model

model = initialize_model()

# Extract features from a single image file
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img).flatten()
        normalized_features = features / norm(features)
        return normalized_features
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# List all valid image files in the directory
image_dir = 'images'
filenames = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]

# Extract features for each image and store them in a list
feature_list = []
for file in tqdm(filenames, desc="Extracting features"):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)

# Save the extracted features and filenames to disk
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print("Feature extraction and saving completed successfully!")



