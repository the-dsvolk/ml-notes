"""
Neural network for house price prediction.
Architecture: 3 inputs → 4 hidden (ReLU) → 1 output (linear).
Uses only Python standard library (math, random).
See neural-networks-by-example.md for the full tutorial.
"""

import random


def relu(x):
    """ReLU(x) = max(0, x)."""
    return x if x > 0 else 0.0


def relu_derivative(x):
    """Derivative of ReLU: 1 if x > 0 else 0."""
    return 1.0 if x > 0 else 0.0


def normalize(value, min_val, max_val):
    """Scale value to [0, 1] using min-max normalization."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def denormalize(norm_value, min_val, max_val):
    """Map [0, 1] back to original scale."""
    return norm_value * (max_val - min_val) + min_val


class NeuralNetwork:
    """
    3 → 4 (ReLU) → 1 network for house price prediction.
    Weights: list of lists. Biases: lists.
    """

    def __init__(self, seed=42):
        random.seed(seed)
        # Input → hidden: 4 neurons, 3 inputs each
        self.weights_input_to_hidden = [
            [random.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(4)
        ]
        self.biases_hidden = [random.uniform(-0.2, 0.2) for _ in range(4)]
        # Hidden → output: 1 neuron, 4 inputs
        self.weights_hidden_to_output = [random.uniform(-0.5, 0.5) for _ in range(4)]
        self.bias_output = random.uniform(-0.2, 0.2)

    def _forward_hidden(self, inputs):
        """Compute hidden layer: z = Wx + b, a = ReLU(z). Returns (pre_activations, activations)."""
        pre_activations = []
        activations = []
        for neuron_idx in range(4):
            weighted_sum = self.biases_hidden[neuron_idx]
            for input_idx in range(3):
                weighted_sum += self.weights_input_to_hidden[neuron_idx][input_idx] * inputs[input_idx]
            pre_activations.append(weighted_sum)
            activations.append(relu(weighted_sum))
        return pre_activations, activations

    def _forward_output(self, hidden_activations):
        """Compute output: single linear neuron."""
        output = self.bias_output
        for i in range(4):
            output += self.weights_hidden_to_output[i] * hidden_activations[i]
        return output

    def forward(self, inputs):
        """
        Full forward pass. inputs: [rooms_norm, area_norm, distance_norm].
        Returns (prediction, hidden_pre_activations, hidden_activations) for use in backprop.
        """
        hidden_pre_activations, hidden_activations = self._forward_hidden(inputs)
        prediction = self._forward_output(hidden_activations)
        return prediction, hidden_pre_activations, hidden_activations

    def backward(self, inputs, hidden_pre_activations, hidden_activations, prediction, actual, learning_rate):
        """
        One step of gradient descent using MSE and chain rule.
        """
        error = prediction - actual

        # Output layer gradients
        grad_weights_output = [error * hidden_activations[i] for i in range(4)]
        grad_bias_output = error

        # Hidden layer: chain rule with ReLU derivative
        grad_weights_hidden = [[0.0] * 3 for _ in range(4)]
        grad_biases_hidden = [0.0] * 4

        for neuron_idx in range(4):
            delta = error * self.weights_hidden_to_output[neuron_idx] * relu_derivative(hidden_pre_activations[neuron_idx])
            grad_biases_hidden[neuron_idx] = delta
            for input_idx in range(3):
                grad_weights_hidden[neuron_idx][input_idx] = delta * inputs[input_idx]

        # Update output weights and bias
        for i in range(4):
            self.weights_hidden_to_output[i] -= learning_rate * grad_weights_output[i]
        self.bias_output -= learning_rate * grad_bias_output

        # Update hidden weights and biases
        for neuron_idx in range(4):
            self.biases_hidden[neuron_idx] -= learning_rate * grad_biases_hidden[neuron_idx]
            for input_idx in range(3):
                self.weights_input_to_hidden[neuron_idx][input_idx] -= learning_rate * grad_weights_hidden[neuron_idx][input_idx]


def train_and_predict():
    # Feature ranges for normalization
    rooms_min, rooms_max = 2, 6
    area_min, area_max = 900, 3000
    dist_min, dist_max = 0.8, 4.0
    price_min, price_max = 150_000, 500_000

    # Toy dataset: [rooms, sqft, distance_km] -> price
    data = [
        ([3, 1500, 2.0], 300_000),
        ([4, 2000, 1.0], 420_000),
        ([2, 1000, 3.0], 220_000),
        ([5, 2500, 0.8], 480_000),
        ([3, 1200, 2.5], 270_000),
        ([4, 1800, 1.5], 380_000),
        ([2, 900, 4.0], 200_000),
        ([5, 2800, 0.5], 510_000),
    ]

    def prepare_sample(features, price):
        rooms, area, dist = features
        x = [
            normalize(rooms, rooms_min, rooms_max),
            normalize(area, area_min, area_max),
            normalize(dist, dist_min, dist_max),
        ]
        y = normalize(price, price_min, price_max)
        return x, y

    # Prepare all samples once (normalization is deterministic; no need to repeat each epoch)
    dataset = [prepare_sample(features, price) for features, price in data]

    nn = NeuralNetwork(seed=42)
    learning_rate = 0.1
    epochs = 500

    print("Training (MSE on normalized target)...")
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs_norm, target_norm in dataset:
            prediction, hidden_pre_activations, hidden_activations = nn.forward(inputs_norm)
            loss = (prediction - target_norm) ** 2
            total_loss += loss
            nn.backward(inputs_norm, hidden_pre_activations, hidden_activations, prediction, target_norm, learning_rate)
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}, mean loss: {total_loss / len(dataset):.6f}")

    # Predict on first house and denormalize
    inputs_test, _ = prepare_sample([3, 1500, 2.0], 300_000)
    prediction_norm, _, _ = nn.forward(inputs_test)
    pred_price = denormalize(prediction_norm, price_min, price_max)
    print(f"\nExample: [3 rooms, 1500 sqft, 2 km] -> predicted ${pred_price:,.0f} (actual $300,000)")

    return nn


if __name__ == "__main__":
    train_and_predict()
