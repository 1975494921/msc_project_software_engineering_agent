# training.py

import numpy as np
from click.core import batch

from model import Model, Tensor
from layers import FullyConnected, ReLU
from utils import cross_entropy_loss, cross_entropy_derivative, softmax
from optimizers import SGD, Adam
from evaluation import evaluate_model

import torch
from torchvision import datasets, transforms


def load_data():
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalization values for MNIST
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the 28x28 images to 784
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Convert the dataset to a DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    # Extract the data and labels
    X_train, Y_train = next(iter(train_loader))

    # Convert labels to one-hot encoding
    Y_train_one_hot = torch.nn.functional.one_hot(Y_train, num_classes=10)

    # Convert to numpy arrays
    X_train = X_train.numpy()
    Y_train_one_hot = Y_train_one_hot.numpy()

    return X_train, Y_train_one_hot


def preprocess_data(X, mean=0.0, std=1.0):
    # Standardization
    return (X - mean) / std


def train_model(model, X_train, Y_train, epochs=10, batch_size=32):
    num_batches = len(X_train) // batch_size
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = Tensor(preprocess_data(X_train[start:end]), requires_grad=False)
            Y_batch = Tensor(Y_train[start:end], requires_grad=False)

            predicted = model.forward(X_batch)
            loss = cross_entropy_loss(Y_batch.data, softmax(predicted.data))
            model.backward(Tensor(cross_entropy_derivative(Y_batch.data, softmax(predicted.data)), requires_grad=False))
            model.optimizer.step()
            epoch_loss += loss

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / num_batches}")


if __name__ == "__main__":
    X_train, Y_train = load_data()

    model = Model()
    model.add(FullyConnected(784, 128))
    model.add(FullyConnected(128, 10))
    model.set_loss_function(cross_entropy_loss)

    # Initialize dummy data to prepare the model (for gradients)
    dummy_x = Tensor(np.random.randn(1, 784), requires_grad=True)
    dummy_y = Tensor(np.zeros((1, 10)), requires_grad=True)
    model.forward(dummy_x)
    dummy_loss = cross_entropy_loss(dummy_y.data, softmax(model.forward(dummy_x).data))
    model.backward(
        Tensor(cross_entropy_derivative(dummy_y.data, softmax(model.forward(dummy_x).data)), requires_grad=False))

    # Set optimizer after initializing gradients
    parameters = []
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            parameters.append(layer.weights)
        if hasattr(layer, 'biases'):
            parameters.append(layer.biases)
    optimizer = Adam(parameters)
    model.set_optimizer(optimizer)

    train_model(model, X_train, Y_train, epochs=5)
    evaluate_model(model, X_train, Y_train)
