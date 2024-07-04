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
sys.path.append('./img_capture')
from img_capture import capture

sys.path.append('./ex1_programs')
from ex1_programs import ex1_model1

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

    while True:
        for file_path in img_folder.glob('*.jpg'):
            if file_path.name not in processed_files:
                processed_files.add(file_path.name)
                new_image = Image(filename=file_path.name, path=str(file_path), camId=camId)
                with app.app_context():
                    db.session.add(new_image)
                    db.session.commit()
                print(f"Inserted {file_path.name} into the database")
        
        time.sleep(10)  # Check every 5 seconds

@app.route('/images', methods=['GET'])
def get_images():
    images = Image.query.all()
    return jsonify([{"id": img.id, "filename": img.filename, "path": img.path} for img in images]), 200

def process_feature_vectors():
    model_dir = os.getenv("MODEL_OUTPUT_DIRECTORY")
    feature_vectors_file = model_dir / 'feature_vectors.csv'

    if feature_vectors_file.exists():
        # Read contents of the file
        with open(feature_vectors_file, 'rb') as file:
            file_data = file.read()

        # Replace with your server endpoint
        server_url = 'http://127.0.0.1:5001'
        
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
    
    time.sleep(5)

if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture.capture_images)
    capture_thread.daemon = True
    capture_thread.start()

    monitor_thread = threading.Thread(target=monitor_folder)
    monitor_thread.daemon = True
    monitor_thread.start()

    monitor_thread = threading.Thread(target=ex1_model1)
    monitor_thread.daemon = True
    monitor_thread.start()
    app.run(debug=True)
    feature_vectors_thread = threading.Thread(target=process_feature_vectors)
    feature_vectors_thread.daemon = True
    feature_vectors_thread.start()


