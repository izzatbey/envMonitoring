import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
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
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU, MaxPooling2D, UpSampling2D
from keras.models import Model
import tensorflow.keras.backend as K
from keras.optimizers import Adam
import numpy as np
import cv2
import time

IMAGE_SIZE = 256

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    #array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), IMAGE_SIZE, IMAGE_SIZE, 3))
    return array

def one_depreprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") * 255.0
    array = np.reshape(array, (IMAGE_SIZE, IMAGE_SIZE, 3))
    return array

def depreprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    #array = array.astype("float32") * 255.0
    array = np.reshape(array, (len(array), IMAGE_SIZE, IMAGE_SIZE, 3))
    return array

#2 各種設定 https://child-programmer.com/ai/cnn-originaldataset-samplecode/#_CNN_8211_ColaboratoryKerasPython

train_data_path = '/home/izzatbey/Documents/KAIT/envMonitoring/data' # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力

image_size = IMAGE_SIZE # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。

color_setting = 3  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。

folder = ['hatake','kawa','mori','tatemono'] # ここを変更。データセット画像のフォルダ名（クラス名）を半角英数で入力

class_number = len(folder)
print('今回のデータで分類するクラス数は「', str(class_number), '」です。')


#3 データセットの読み込みとデータ形式の設定・正規化・分割 

X_image = []  
Y_label = []

for index, name in enumerate(folder):
  read_data = train_data_path + '/' + name
  files = glob.glob(read_data + '/*.png') #ここを変更。png形式のファイルを利用する場合のサンプルです。
  print('--- 読み込んだデータセットは', read_data, 'です。')
  num=0
  for i, file in enumerate(files):
    if color_setting == 1:
      img = load_img(file, color_mode = 'grayscale' ,target_size=(image_size, image_size))  
    elif color_setting == 3:
      img = load_img(file, color_mode = 'rgb' ,target_size=(image_size, image_size))
    array = img_to_array(img)
    X_image.append(array)
    num +=1
    Y_label.append(index)
  print('index: ',index,' num:',num)

X_image = np.array(X_image)
Y_label = np.array(Y_label)

X_image = X_image.astype('float32') / 255
print(len(X_image))
#↓ここの部分をコメントアウトする
#Y_label = keras.utils.to_categorical(Y_label, class_number) #Kerasのバージョンなどにより使えないのでコメントアウト
#Y_label = np_utils.to_categorical(Y_label, class_number) #上記のコードのかわり

train_images, valid_images ,train_labels ,valid_labels = train_test_split(X_image,Y_label,test_size=0.20,shuffle = True)
x_train = train_images
y_train = train_labels
x_test = valid_images
y_test = valid_labels

# n=5
# for i in range(10):
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_train[[i]].reshape(image_size,image_size,3))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.set_title(str(y_train[i]))
# plt.show()

LEARNING_RATE = 0.0004
BATCH_SIZE = 64
Z_DIM = 1000
EPOCHS = 3

encoder_input = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3), name='encoder_input')
x = encoder_input
x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', name='encoder_conv_0')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', name='encoder_conv_0_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', name='encoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_3')(x)
x = LeakyReLU()(x)
#encoder_output2=Dense(Z_DIM, name='encoder_output')(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
encoder_output = Dense(Z_DIM, name='encoder_output')(x)
encoder = Model(encoder_input, encoder_output)

# デコーダ
decoder_input = Input(shape=(Z_DIM,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_0')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2_6')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_3')(x)
x = Activation('sigmoid')(x)
decoder_output = x
decoder = Model(decoder_input, decoder_output)


# エンコーダ/デコーダ連結
model_input = encoder_input
model_output = decoder(encoder_output)
model = Model(model_input, model_output)

# 学習用設定設定（最適化関数、損失関数）
optimizer = Adam(learning_rate=LEARNING_RATE)

def r_loss(y_true, y_pred):
  return K.mean(K.square(y_true - y_pred), axis=[1,2,3])

model.compile(optimizer=optimizer, loss=r_loss, metrics=['accuracy'])

# 学習実行
history=model.fit(
    x_train,
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, x_test),
)

model.summary()

model_dir = "/home/izzatbey/Documents/KAIT/envMonitoring/model"
model_path = os.path.join(model_dir, "ex1_model1{}.keras".format(Z_DIM))
os.makedirs(model_dir, exist_ok=True)
model.save(model_path)

model_path_load = "/home/izzatbey/Documents/KAIT/envMonitoring/model/ex1_model11000.keras"
model = load_model(model_path_load, custom_objects={"r_loss": r_loss})

encoder_output_layer = model.get_layer('encoder_output').output
encoder_input_layer = model.input
encoder = Model(encoder_input_layer, encoder_output_layer)

feature_vectors = encoder.predict(x_train)
print("Feature Vectors:", feature_vectors)
print("Feature Vectors Shape:", feature_vectors.shape)
saved_vectors = np.array(feature_vectors)
np.savetxt("/home/izzatbey/Documents/KAIT/envMonitoring/feature_vectors.csv", saved_vectors, delimiter=',')
print("Feature vectors Saved!")
if os.path.exists("/home/izzatbey/Documents/KAIT/envMonitoring/model/ex1_model11000.keras"):
   os.remove("/home/izzatbey/Documents/KAIT/envMonitoring/model/ex1_model11000.keras")
   print(".keras file successfully deleted.")
else:
   print(".keras file does not exist. File deletion skipped")
# loaded_feature_vectors = np.loadtxt("/home/izzatbey/Documents/KAIT/env_monitoring/feature_vectors.csv", delimiter=',')
# print(f"Loaded feature vectors shape: {loaded_feature_vectors}")
time.sleep(5)