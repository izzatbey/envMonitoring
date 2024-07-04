import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from keras import layers

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
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import cv2


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

train_data_path = 'F:/国土地理院/26/data' # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力

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

Z_DIM=500

def r_loss(y_true, y_pred):
  return K.mean(K.square(y_true - y_pred), axis=[1,2,3])

Encoder_model = keras.models.load_model("./model/AE_{}".format(Z_DIM), custom_objects={"r_loss": r_loss })
Encoder_model.summary()


#Metric_model = keras.models.load_model("./model/ML1")
#Metric_model.summary()

layer_name = 'encoder_output'
intermediate_layer_model = Model(inputs=Encoder_model.input,
                                 outputs=Encoder_model.get_layer(layer_name).output)


x_train2 = intermediate_layer_model.predict(x_train)
print(x_train2.shape)
x_test2 = intermediate_layer_model.predict(x_test)
print(x_test2.shape)


#######################################
#y_train = np.squeeze(y_train)
#y_test = np.squeeze(y_test)


# Show a collage of 5x5 random images.
#sample_idxs = np.random.randint(0, 100, size=(5, 5))
#examples = x_train[sample_idxs]
examples = x_train2
#show_collage(examples)

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    #print(y_train)
    #print(y_train.shape)
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)


num_classes = 4

class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batchs):
        self.num_batchs = num_batchs

    def __len__(self):
        return self.num_batchs

    def __getitem__(self, _idx):
        #x = np.empty((2, num_classes, 64,64,50), dtype=np.float32)
        x = np.empty((2, num_classes, Z_DIM), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = x_train2[anchor_idx]
            x[1, class_idx] = x_train2[positive_idx]
        return x
    
#examples = next(iter(AnchorPositivePairs(num_batchs=1)))

#show_collage(examples)

class EmbeddingModel(keras.Model):
    #train_step(self, data) メソッドだけをオーバーライドします。
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.未解決の問題の回避策。削除される予定です。
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            # モデルを通してアンカーとポジティブの両方を実行します。
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            # アンカーとポジティブの間のコサイン類似度を計算します。彼らがそうしているように
            # 正規化されているため、これは単なるペアごとの内積です。
            #　ランダムに選択されたものとその対応する同ラベルとの内積
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            # これらをロジットとして使用するつもりなので、温度によってスケールします。
            # この値は通常、ハイパーパラメータとして選択されます。
            temperature = 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            # これらの類似性をソフトマックスのロジットとして使用します。のラベル
            # この呼び出しはシーケンス [0, 1, 2, ..., num_classes] です。
            # アンカー/ポジティブに対応する主な対角値が必要です
            # ペア、高くなります。この損失により、エンベディングが移動します。
            # アンカー/ポジティブペアを一緒に固定し、他のすべてのペアを離します。
            sparse_labels = tf.range(num_classes)
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        #勾配を計算し、オプティマイザーを介して適用します。
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        #メトリクス (特に損失値のメトリクス) を更新して返します。
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}



inputs2 = layers.Input(shape=(Z_DIM))
x2 = layers.Dense(units=1000, activation='relu')(inputs2)
x2 = layers.Dense(units=2000, activation='relu')(x2)
embeddings = layers.Dense(units=1000, activation=None)(x2)
embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

model2 = EmbeddingModel(inputs2, embeddings)


model2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-3),
    #one-hot 表現でラベルが作成されている場合は CategoricalCrossentropy を利用する
    #整数でラベルが作成されている場合は、SparseCategoricalCrossentropy を利用する
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model2.fit(AnchorPositivePairs(num_batchs=64), epochs=100)
model2.summary()
model2.save("./model/ex1_model3_{}".format(Z_DIM))

