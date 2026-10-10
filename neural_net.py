"""
Neural Network from Scratch — PyBlitz
======================================
A tiny neural network (a single neuron) built in pure Python.
No TensorFlow, no PyTorch, no ML libraries — just math.

It LEARNS the AND logic gate:
    [0, 0] -> 0
    [0, 1] -> 0
    [1, 0] -> 0
    [1, 1] -> 1

How it works:
  1. weights + bias start as random numbers
  2. sigmoid() squashes any number into the 0..1 range
  3. predict() = sigmoid(weights x inputs + bias)
  4. training loop runs 1000 epochs: each round it predicts,
     measures the error, and nudges the weights/bias to shrink it
  5. final predictions: all 4 correct!

Run:  python neural_net.py
"""
import math
import random

random.seed(7)

# Training data: (inputs, target) for the AND gate
data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

# Random starting weights + bias
w = [random.uniform(-1, 1) for _ in range(2)]
b = random.uniform(-1, 1)


def sig(x):
    """Sigmoid activation: squashes any number into 0..1."""
    return 1 / (1 + math.exp(-x))


def predict(i):
    """One neuron: weighted sum of inputs + bias, through sigmoid."""
    total = 0
    for x, y in zip(i, w):
        total += x * y
    return sig(total + b)


# Training: 1000 epochs of "predict -> measure error -> adjust"
for _ in range(1000):
    for i, t in data:
        o = predict(i)
        e = t - o                      # the error
        for j in range(2):
            w[j] += 0.5 * e * i[j]     # nudge weights
        b += 0.5 * e                   # nudge bias

print("Training complete! Predictions:")
for i, t in data:
    p = 1 if predict(i) >= 0.5 else 0
    mark = "OK" if p == t else "WRONG"
    print(f"  Inputs {i} -> Prediction: {p} | Target: {t} | {mark}")
