import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize the model
def initialize_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
 
    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model

model = initialize_model()
st.title('Deep Learning In Ecommerce')

st.title('Fashion Recommender System')

# Save uploaded file to the 'uploads' directory
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Extract features from an image file using the model
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img).flatten()
    normalized_features = features / norm(features)
    return normalized_features

# Recommend similar images based on extracted features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload -> save -> extract features -> recommend -> display
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image')

        # Extract features
        features = feature_extraction(file_path, model)

        # Get recommendations
        indices = recommend(features, feature_list)

        # Display recommended images
        st.write("Recommended Products:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            col.image(filenames[indices[0][i]])
    else:
        st.error("An error occurred during file upload.")
 