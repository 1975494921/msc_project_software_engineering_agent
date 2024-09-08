# evaluation.py

import numpy as np
from model import Model, Tensor
from layers import FullyConnected
from utils import softmax

def accuracy(y_true, y_pred):
    correct = np.sum(y_true.argmax(axis=1) == y_pred.argmax(axis=1))
    return correct / y_true.shape[0]

def precision_recall_fscore(y_true, y_pred, average='macro'):
    epsilon = 1e-7
    true_positives = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    predicted_positives = np.sum(y_pred == 1, axis=0)
    actual_positives = np.sum(y_true == 1, axis=0)
    
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (actual_positives + epsilon)
    fscore = 2 * (precision * recall) / (precision + recall + epsilon)
    
    if average == 'macro':
        precision = np.mean(precision)
        recall = np.mean(recall)
        fscore = np.mean(fscore)
    
    return precision, recall, fscore

def evaluate_model(model, X_test, Y_test):
    predictions = model.predict(Tensor(X_test)).data
    predictions = softmax(predictions)

    acc = accuracy(Y_test, predictions)
    prec, rec, fscore = precision_recall_fscore(Y_test, predictions)

    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F-Score: {fscore}")

if __name__ == "__main__":
    # Assuming X_test and Y_test are loaded similarly to training data in training.py
    X_test = np.random.randn(200, 784)  # Example test data
    Y_test = np.zeros((200, 10))
    Y_test[np.arange(200), np.random.randint(0, 10, 200)] = 1  # Random test labels for 10 classes
    
    model = Model()
    model.add(FullyConnected(784, 128))
    model.add(FullyConnected(128, 10))
    
    # Load model weights if saved, or assume the model is trained and weights are initialized
    
    evaluate_model(model, X_test, Y_test)
