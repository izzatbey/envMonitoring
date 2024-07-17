FROM nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3

# Set the working directory in the container
WORKDIR /usr/src/app

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev \
    liblapack-dev libblas-dev gfortran \
    python3-pip libpq-dev libgl1-mesa-glx \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran

# Ensure HDF5 library paths are set correctly
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/hdf5/serial/

# Install necessary Python packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install -U numpy

# Install OpenCV
RUN pip3 install opencv-python-headless

# Install other dependencies listed in requirements-jetson.txt
COPY requirements-jetson.txt .
RUN pip3 install -r requirements-jetson.txt

# Copy the entire current directory into the container
COPY . .

# Set the entry point for the container
ENTRYPOINT ["python3", "app.py"]
