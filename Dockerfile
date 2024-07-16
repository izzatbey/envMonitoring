FROM nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3

# Set the working directory in the container
WORKDIR /usr/src/app

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev \
    liblapack-dev libblas-dev gfortran \
    python3-pip libpq-dev libgl1-mesa-glx

# Ensure HDF5 library paths are set correctly
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/hdf5/serial/

# Install necessary Python packages
#RUN python3 -m pip install --upgrade pip && \
#    pip3 install -U testresources setuptools && \
#    pip3 install -U numpy future mock \
#    keras_preprocessing keras_applications \
#    gast protobuf pybind11 cython pkgconfig \
#    packaging h5py

# Install TensorFlow for NVIDIA Jetson
#RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 tensorflow==2.12.0+nv23.06

# Install other dependencies listed in requirements-jetson.txt
COPY requirements-jetson.txt .
RUN pip3 install -r requirements-jetson.txt

# Copy the entire current directory into the container
COPY . .

# Set the entry point for the container
ENTRYPOINT ["python3", "app.py"]
