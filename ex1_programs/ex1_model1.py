from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image

from tensorflow import keras
from keras import layers
import os
from tensorflow.keras.models import load_model
from keras.datasets import cifar10
import keras
from keras import layers
from PIL import Image, ImageFilter
from keras.datasets import mnist
import numpy as np
import glob

from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU, MaxPooling2D, UpSampling2D
from keras.models import Model
import tensorflow.keras.backend as K
from keras.optimizers import Adam
import numpy as np
import cv2
import time
from dotenv import load_dotenv
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import ModelCheckpoint

# Enable mixed precision
set_global_policy('mixed_float16')

# Set environment variable for memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

load_dotenv()

IMAGE_SIZE = 128  # Reduce image size to decrease memory load

def preprocess(array):
    array = np.reshape(array, (len(array), IMAGE_SIZE, IMAGE_SIZE, 3))
    return array / 255.0

def depreprocess(array):
    array = np.reshape(array, (len(array), IMAGE_SIZE, IMAGE_SIZE, 3))
    return array * 255.0

def run_model():
    train_data_path = os.getenv("TRAIN_DATA_DIRECTORY")
    image_size = IMAGE_SIZE
    color_setting = 3
    folder = ['hatake', 'kawa', 'mori', 'tatemono']
    class_number = len(folder)
    print('今回のデータで分類するクラス数は「', str(class_number), '」です。')

    X_image = []
    Y_label = []

    for index, name in enumerate(folder):
        read_data = train_data_path + '/' + name
        files = glob.glob(read_data + '/*.png')
        print('--- 読み込んだデータセットは', read_data, 'です。')
        num = 0
        for i, file in enumerate(files):
            if color_setting == 1:
                img = load_img(file, color_mode='grayscale', target_size=(image_size, image_size))
            elif color_setting == 3:
                img = load_img(file, color_mode='rgb', target_size=(image_size, image_size))
            array = img_to_array(img)
            X_image.append(array)
            num += 1
            Y_label.append(index)
        print('index: ', index, ' num:', num)

    X_image = np.array(X_image)
    Y_label = np.array(Y_label)

    X_image = X_image.astype('float32') / 255
    print(len(X_image))

    train_images, valid_images, train_labels, valid_labels = train_test_split(X_image, Y_label, test_size=0.20, shuffle=True)
    x_train = train_images
    y_train = train_labels
    x_test = valid_images
    y_test = valid_labels

    LEARNING_RATE = 0.0004
    BATCH_SIZE = 16
    Z_DIM = 256
    EPOCHS = 3

    # Clear TensorFlow session
    K.clear_session()

    encoder_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='encoder_input')
    x = encoder_input
    x = Conv2D(filters=8, kernel_size=3, strides=1, padding='same', name='encoder_conv_0')(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', name='encoder_conv_0_1')(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', name='encoder_conv_1')(x)
    x = LeakyReLU()(x)
    shape_before_flattening = K.int_shape(x)[1:]
    x = Flatten()(x)
    encoder_output = Dense(Z_DIM, name='encoder_output')(x)
    encoder = Model(encoder_input, encoder_output)

    decoder_input = Input(shape=(Z_DIM,), name='decoder_input')
    x = Dense(np.prod(shape_before_flattening))(decoder_input)
    x = Reshape(shape_before_flattening)(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2')(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2_6')(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(filters=8, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_3')(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_4')(x)
    x = Activation('sigmoid')(x)
    decoder_output = x
    decoder = Model(decoder_input, decoder_output)

    model_input = encoder_input
    model_output = decoder(encoder_output)
    model = Model(model_input, model_output)

    optimizer = Adam(learning_rate=LEARNING_RATE)

    def r_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

    model.compile(optimizer=optimizer, loss=r_loss, metrics=['accuracy'])

    checkpoint = ModelCheckpoint('model_checkpoint.h5', save_best_only=True, monitor='val_loss')

    history = model.fit(
        x_train,
        x_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, x_test),
        callbacks=[checkpoint]
    )

    model.summary()

    model_dir = os.getenv("MODEL_OUTPUT_DIRECTORY")
    model_path = os.path.join(model_dir, f"ex1_model1{Z_DIM}.keras")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)

    model_path_load = os.path.join(model_dir, f"ex1_model1{Z_DIM}.keras")
    model = load_model(model_path_load, custom_objects={"r_loss": r_loss})

    encoder_output_layer = model.get_layer('encoder_output').output
    encoder_input_layer = model.input
    encoder = Model(encoder_input_layer, encoder_output_layer)

    feature_vectors = encoder.predict(x_train)
    print("Feature Vectors:", feature_vectors)
    print("Feature Vectors Shape:", feature_vectors.shape)
    saved_vectors = np.array(feature_vectors)
    np.savetxt(os.path.join(model_dir, "feature_vectors.csv"), saved_vectors, delimiter=',')
    print("Feature vectors Saved!")
    
    if os.path.exists(model_path_load):
        os.remove(model_path_load)
        print(".keras file successfully deleted.")
    else:
        print(".keras file does not exist. File deletion skipped")

    start_time = time.time()
    while time.time() - start_time < 30:
        time.sleep(10)  # Check every 10 seconds if it's time to exit
    print("Exiting run_model function.")
