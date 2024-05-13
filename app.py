import tensorflow as tf
import sys
import os

# For creating sample model, can remove if bringing your own model
import mnist 


def main(argc: int, argv: list[str]) -> int:
    model_path = os.getenv('KERAS_MODEL_PATH')
    if model_path is None:
        # Create and use MNIST sample model is no model is specified to load, can remove if bringing your own model
        model_path = 'mnist.keras'
        mnist.create_mnist_model()

    model = tf.keras.models.load_model(model_path) 

    # Show the model architecture
    model.summary()

    return 0

if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
