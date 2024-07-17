import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import pandas as pd
import json
import requests
from dotenv import load_dotenv
import time


load_dotenv()

def load_images_from_subfolders(main_folder, subfolders, image_size=(128, 128), log_file="image_log.txt"):
    images = []
    image_names = []

    with open(log_file, 'w') as log:
        for subfolder in subfolders:
            folder_path = os.path.join(main_folder, subfolder)
            if not os.path.exists(folder_path):
                print(f"Warning: Subfolder {folder_path} does not exist.")
                continue
            print(f"Accessing folder: {folder_path}")  # Debug statement
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                print(f"Loading image: {img_path}")  # Debug statement
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    image_names.append(img_path)
                    log.write(f"{img_path}\n")
                else:
                    print(f"Failed to load image: {img_path}")  # Debug statement
    return np.array(images), image_names


def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder

def train_and_save_model(train_data_path, subfolders, model_path='autoencoder.h5', log_file="image_log.txt"):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                # Check if virtual devices are configured
                if not tf.config.experimental.get_virtual_device_configuration(gpu):
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    while True:
        images, _ = load_images_from_subfolders(train_data_path, subfolders, log_file=log_file)
        print(f"Total images loaded: {len(images)}")  # Debug statement
        
        if len(images) >= 50:
            X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)

            input_shape = (128, 128, 3)
            autoencoder, encoder = build_autoencoder(input_shape)
            autoencoder.fit(X_train, X_train, epochs=3, batch_size=16, validation_data=(X_test, X_test))

            autoencoder.save(model_path)
            encoder.save(model_path.replace('.h5', '_encoder.h5'))
            
            print("Model training completed and saved.")
            break  # Exit the loop after successful training
        else:
            print("Not enough images to proceed with training. Sleeping for 30 seconds.")
            time.sleep(30)

def load_model_and_predict(model_path, images):
    encoder = load_model(model_path)
    encoded_images = encoder.predict(images)
    
    n_samples = encoded_images.shape[0]
    flattened_features = encoded_images.reshape(n_samples, -1)
    
    return flattened_features

def send_file_to_server(features_file, zip_file, server_url, max_retries=5, sleep_time=30):
    for attempt in range(max_retries):
        try:
            with open(features_file, 'rb') as ff, open(zip_file, 'rb') as zf:
                files = {'feature_vectors': ff, 'raw_images': zf}
                response = requests.post(server_url, files=files)
                response.raise_for_status()
            print(f"Files sent successfully on attempt {attempt + 1}")
            return response.status_code
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Files could not be sent.")
    return None

def delete_images(image_names):
    for img_path in image_names:
        try:
            os.remove(img_path)
            print(f"Deleted {img_path}")
        except OSError as e:
            print(f"Error deleting {img_path}: {e}")
