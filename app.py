from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from pathlib import Path
from dotenv import load_dotenv
import os
import threading
import time
import sys
import requests
import sqlalchemy
import pandas as pd

sys.path.append('./img_capture')
from img_capture import capture

sys.path.append('./ex1_programs')
from new_model_izzat import train_and_save_model, load_model_and_predict, send_file_to_server, delete_images, load_images_from_subfolders
load_dotenv()

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), unique=True, nullable=False)
    path = db.Column(db.String(200), nullable=False)
    camId = db.Column(db.String(120), nullable=False)

    def __init__(self, filename, path, camId):
        self.filename = filename
        self.path = path
        self.camId = camId

with app.app_context():
    db.create_all()

def monitor_folder():
    processed_files = set()
    env_img_folder = os.getenv("IMG_FOLDER")
    img_folder = Path(env_img_folder)
    camId = os.getenv("CAM_ID")

    # while True:
    for file_path in img_folder.glob('*.jpg'):
        if file_path.name not in processed_files:
            new_image = Image(filename=file_path.name, path=str(file_path), camId=camId)
            with app.app_context():
                try:
                    db.session.add(new_image)
                    db.session.commit()
                    processed_files.add(file_path.name)
                    print(f"Inserted {file_path.name} into the database")
                except sqlalchemy.exc.IntegrityError:
                    db.session.rollback()
                    print(f"Duplicate entry {file_path.name}, skipping.")
        
        time.sleep(10)  # Check every 10 seconds

@app.route('/images', methods=['GET'])
def get_images():
    images = Image.query.all()
    return jsonify([{"id": img.id, "filename": img.filename, "path": img.path} for img in images]), 200

def process_feature_vectors():
    model_dir = Path(os.getenv("MODEL_OUTPUT_DIRECTORY"))
    # while True:
    feature_vectors_file = model_dir / 'feature_vectors.csv'

    if feature_vectors_file.exists():
        # Read contents of the file
        with open(feature_vectors_file, 'rb') as file:
            file_data = file.read()

        # Replace with your server endpoint
        server_url = os.getenv("SERVER_UPLOAD_URL")
            
        try:
            # Send file to server using POST request
            response = requests.post(server_url, files={'file': file_data})
            response.raise_for_status()  # Raise an exception for HTTP errors
                
            # If upload successful, delete the file
            os.remove(feature_vectors_file)
            print(f"Uploaded and deleted {feature_vectors_file}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error uploading file: {e}")

    else:
        print(f"{feature_vectors_file} does not exist.")
        
    time.sleep(30)  # Sleep for 30 seconds before next iteration

def model_thread_function(train_data_path, subfolders, model_path, server_url, log_file):
    model_time_start = time.time()
    # Train and save the model
    train_and_save_model(train_data_path, subfolders, model_path, log_file)

    # Load images for prediction (this can be modified as per your requirements)
    images, image_names = load_images_from_subfolders(train_data_path, subfolders, log_file=log_file)

    # Load model and predict
    features = load_model_and_predict(model_path.replace('.h5', '_encoder.h5'), images)

    # Save features to CSV
    features_file = "encoded_features.csv"
    pd.DataFrame(features).to_csv(features_file, index=False)

    model_time_end = time.time()
    execution_time = model_time_end - model_time_start
    print("Execution Model Time : ", execution_time)

    # Send features file to server
    if server_url:
        status_code = send_file_to_server(features_file, server_url)
        if status_code == 200:
            print("Data sent successfully to the server.")
            delete_images(image_names)
            os.remove(features_file)
        else:
            print(f"Failed to send data to server. Status code: {status_code}")
    else:
        print("SERVER_URL environment variable is not set. Skipping data send.")
    



if __name__ == '__main__':
    train_data_path = os.getenv("TRAIN_DATA_DIRECTORY")
    if train_data_path is None:
        raise ValueError("TRAIN_DATA_DIRECTORY environment variable is not set. Please set it to the path of your image folder.")
    
    subfolders = ['camera']
    model_path = "autoencoder.h5"
    server_url = os.getenv("SERVER_URL")
    log_file = "image_log.txt"

    model_thread_function(train_data_path, subfolders, model_path, server_url, log_file)
    # Start the model thread
    # model_thread = threading.Thread(target=model_thread_function, args=(train_data_path, subfolders, model_path, server_url, log_file))
    # model_thread.daemon = True
    # model_thread.start()

    # capture_thread = threading.Thread(target=capture.capture_images)
    # capture_thread.daemon = True
    # capture_thread.start()

    app.run(debug=True)
