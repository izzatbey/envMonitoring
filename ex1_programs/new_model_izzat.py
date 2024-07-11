import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
import pandas as pd
import json
import requests
from dotenv import load_dotenv

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
    images, _ = load_images_from_subfolders(train_data_path, subfolders, log_file=log_file)
    print(f"Total images loaded: {len(images)}")  # Debug statement
    if len(images) == 0:
        raise ValueError("No images were loaded. Please check the data directory and subfolders.")
    X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)

    input_shape = (128, 128, 3)
    autoencoder, encoder = build_autoencoder(input_shape)
    autoencoder.fit(X_train, X_train, epochs=3, batch_size=32, validation_data=(X_test, X_test))

    autoencoder.save(model_path)
    encoder.save(model_path.replace('.h5', '_encoder.h5'))

def load_model_and_predict(model_path, images):
    encoder = load_model(model_path)
    encoded_images = encoder.predict(images)
    
    n_samples = encoded_images.shape[0]
    flattened_features = encoded_images.reshape(n_samples, -1)
    
    return flattened_features

def send_file_to_server(file_path, server_url):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(server_url, files=files)
    return response.status_code

def delete_images(image_names):
    for img_path in image_names:
        try:
            os.remove(img_path)
            print(f"Deleted {img_path}")
        except OSError as e:
            print(f"Error deleting {img_path}: {e}")
