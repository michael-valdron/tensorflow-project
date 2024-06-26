import tensorflow as tf

class MNISTDataset:
    def __init__(self) -> None:
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

def create_mnist_model():
    dataset = MNISTDataset()
    dataset.load_data()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])
    model.fit(dataset.x_train, dataset.y_train, epochs=5)
    model.save('mnist.keras')
