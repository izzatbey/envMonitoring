import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers
from collections import defaultdict
import matplotlib.pyplot as plt

# Load the pre-trained encoder model
def load_encoder_model(model_path):
    Encoder_model = keras.models.load_model(model_path, custom_objects={"r_loss": r_loss})
    Encoder_model.summary()
    return Encoder_model

# Extract intermediate layer outputs
def extract_features(model, x_train, x_test, layer_name='final_encoding'):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    x_train_features = intermediate_layer_model.predict(x_train)
    x_test_features = intermediate_layer_model.predict(x_test)
    return x_train_features, x_test_features

# Prepare data for training
def prepare_data(x_train_features, x_test_features, y_train, y_test, num_classes):
    class_idx_to_train_idxs = defaultdict(list)
    for y_train_idx, y in enumerate(y_train):
        class_idx_to_train_idxs[y].append(y_train_idx)

    class_idx_to_test_idxs = defaultdict(list)
    for y_test_idx, y in enumerate(y_test):
        class_idx_to_test_idxs[y].append(y_test_idx)

    return class_idx_to_train_idxs, class_idx_to_test_idxs

# Main class for the embedding model
class EmbeddingModel(keras.Model):
    def __init__(self, num_classes, x_train_features):
        super(EmbeddingModel, self).__init__()
        self.num_classes = num_classes
        self.x_train_features = x_train_features
        self.Z_DIM = x_train_features.shape[1]

    def build(self, input_shape):
        self.inputs = layers.Input(shape=(self.Z_DIM,))
        self.embeddings = layers.Dense(units=8, activation=None)(self.inputs)
        self.embeddings = tf.nn.l2_normalize(self.embeddings, axis=-1)
        self.model = keras.Model(self.inputs, self.embeddings)
        self.model.summary()

    def compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            anchor_embeddings = self.model(anchors, training=True)
            positive_embeddings = self.model(positives, training=True)
            similarities = tf.einsum("ae,pe->ap", anchor_embeddings, positive_embeddings)
            temperature = 0.2
            similarities /= temperature
            sparse_labels = tf.range(self.num_classes)
            loss = self.compiled_loss(sparse_labels, similarities)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}

    def train_model(self, num_batches, epochs, class_idx_to_train_idxs):
        class AnchorPositivePairs(keras.utils.Sequence):
            def __init__(self, num_batches):
                self.num_batches = num_batches

            def __len__(self):
                return self.num_batches

            def __getitem__(self, _idx):
                x = np.empty((2, self.num_classes, self.Z_DIM), dtype=np.float32)
                for class_idx in range(self.num_classes):
                    examples_for_class = class_idx_to_train_idxs[class_idx]
                    anchor_idx = random.choice(examples_for_class)
                    positive_idx = random.choice(examples_for_class)
                    while positive_idx == anchor_idx:
                        positive_idx = random.choice(examples_for_class)
                    x[0, class_idx] = self.x_train_features[anchor_idx]
                    x[1, class_idx] = self.x_train_features[positive_idx]
                return x

        history = self.model.fit(AnchorPositivePairs(num_batches), epochs=epochs)
        self.model.save("./model/ML1")
        return history

    def plot_history(self, history):
        plt.plot(history.history["loss"])
        plt.show()

    def evaluate_model(self, x_test_features, y_test, num_collage_examples=5, near_neighbours_per_example=4):
        embeddings = self.model.predict(x_test_features)
        gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
        near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1):]

        examples = np.empty(
            (
                num_collage_examples,
                near_neighbours_per_example + 1,
                self.Z_DIM,
            ),
            dtype=np.float32,
        )
        for row_idx in range(num_collage_examples):
            examples[row_idx, 0] = x_test_features[row_idx]
            anchor_near_neighbours = reversed(near_neighbours[row_idx][:-1])
            for col_idx, nn_idx in enumerate(anchor_near_neighbours):
                examples[row_idx, col_idx + 1] = x_test_features[nn_idx]

        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for class_idx in range(self.num_classes):
            example_idxs = class_idx_to_test_idxs[class_idx][:10]
            for y_test_idx in example_idxs:
                for nn_idx in near_neighbours[y_test_idx][:-1]:
                    nn_class_idx = y_test[nn_idx]
                    confusion_matrix[class_idx, nn_class_idx] += 1

        return confusion_matrix

# Example usage
model_path = "/home/takanolab/proglams_python/model/ae_encoding1"
encoder_model = load_encoder_model(model_path)
x_train2, x_test2 = extract_features(encoder_model, x_train, x_test)

num_classes = 4
class_idx_to_train_idxs, class_idx_to_test_idxs = prepare_data(x_train2, x_test2, y_train, y_test, num_classes)

embedding_model = EmbeddingModel(num_classes, x_train2)
embedding_model.build(input_shape=(x_train2.shape[1],))
embedding_model.compile_model()

history = embedding_model.train_model(num_batches=50, epochs=300, class_idx_to_train_idxs=class_idx_to_train_idxs)
embedding_model.plot_history(history)

confusion_matrix = embedding_model.evaluate_model(x_test2, y_test)
print("Confusion Matrix:\n", confusion_matrix)
