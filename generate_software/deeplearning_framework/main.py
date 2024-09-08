import numpy as np
from model import Model, Tensor
from layers import FullyConnected, ReLU
from optimizers import SGD, Adam
from training import train_model, load_data
from evaluation import evaluate_model
from cnn_layers import Convolution, MaxPooling
from rnn_layers import RNNCell
from lstm_layers import LSTMCell
from transformer_layers import TransformerBlock
from utils import cross_entropy_loss, cross_entropy_derivative, softmax


def create_mlp_model(input_dim, output_dim):
    model = Model()
    model.add(FullyConnected(input_dim, 128))
    model.add(ReLU())
    model.add(FullyConnected(128, 64))
    model.add(ReLU())
    model.add(FullyConnected(64, output_dim))
    return model


def create_cnn_model():
    model = Model()
    model.add(Convolution(1, 16, 5, stride=1, pad=2))  # Assuming input image is 1x28x28 (MNIST)
    model.add(MaxPooling(2, stride=2))
    model.add(Convolution(16, 32, 5, stride=1, pad=2))
    model.add(MaxPooling(2, stride=2))
    model.add(FullyConnected(32 * 7 * 7, 128))
    model.add(FullyConnected(128, 10))
    return model


def create_rnn_model(input_dim, hidden_dim, output_dim):
    model = Model()
    model.add(RNNCell(input_dim, hidden_dim))
    model.add(FullyConnected(hidden_dim, output_dim))
    return model


def create_lstm_model(input_dim, hidden_dim, output_dim):
    model = Model()
    model.add(LSTMCell(input_dim, hidden_dim))
    model.add(FullyConnected(hidden_dim, output_dim))
    return model


def create_transformer_model():
    model = Model()
    model.add(TransformerBlock(num_heads=8, model_dim=512))
    model.add(FullyConnected(512, 10))  # Assuming a classification task
    return model


if __name__ == "__main__":
    # Load and preprocess data
    X_train, Y_train = load_data()

    # Select model type
    model_type = "mlp"  # Options: "mlp", "cnn", "rnn", "lstm", "transformer"

    if model_type == "mlp":
        model = create_mlp_model(784, 10)
    elif model_type == "cnn":
        model = create_cnn_model()
    elif model_type == "rnn":
        model = create_rnn_model(784, 128, 10)  # Example dimensions
    elif model_type == "lstm":
        model = create_lstm_model(784, 128, 10)  # Example dimensions
    elif model_type == "transformer":
        model = create_transformer_model()
    else:
        raise ValueError("Unknown model type")

    # Initialize dummy data to prepare the model (for gradients)
    dummy_x = Tensor(np.random.randn(1, 784), requires_grad=True)
    dummy_y = Tensor(np.zeros((1, 10)), requires_grad=True)
    model.forward(dummy_x)
    model.forward(dummy_x)
    dummy_loss = cross_entropy_loss(dummy_y.data, softmax(model.forward(dummy_x).data))
    model.backward(
        Tensor(cross_entropy_derivative(dummy_y.data, softmax(model.forward(dummy_x).data)), requires_grad=False))

    # Define loss and optimizer
    model.set_loss_function(cross_entropy_loss)
    parameters = [layer.weights for layer in model.layers if hasattr(layer, 'weights')]
    parameters += [layer.biases for layer in model.layers if hasattr(layer, 'biases')]
    model.set_optimizer(Adam(parameters))

    # Train model
    train_model(model, X_train, Y_train, epochs=5)

    # Evaluate model
    X_test = np.random.randn(200, 784)  # Dummy test data
    Y_test = np.zeros((200, 10))
    Y_test[np.arange(200), np.random.randint(0, 10, 200)] = 1  # Dummy test labels for 10 classes

    evaluate_model(model, X_test, Y_test)
