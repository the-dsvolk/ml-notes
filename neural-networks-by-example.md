# Neural Networks by Example: House Price Prediction

A step-by-step tutorial that builds intuition with a real-estate example, then implements a small neural network in plain Python—no PyTorch or NumPy.

---

## Part 1: Understanding the Basics

### 1.1 What Problem Are We Solving?

Imagine you're a real estate agent. A client asks: *"What should this house cost?"*

You look at:

- **Rooms:** 3 bedrooms  
- **Size:** 1,500 square feet  
- **Location:** 2 km from the market  

Your brain has seen hundreds of houses and learned patterns. You think: *"Similar houses sold for around $300,000."*

That's what we're teaching the computer to do—**learn patterns from data**.

---

### 1.2 Why Can't We Just Use a Simple Formula?

You might think: just make a formula!

```
Price = (rooms × 50000) + (sqft × 100) + (market_distance × -20000)
```

**Problem:** Real estate doesn't work that simply.

- A 5-bedroom house in a bad location isn't better than a 3-bedroom in a great location  
- Square footage matters more for big houses than small ones  
- Features **interact** in complex ways  

We need something that can learn these **non-linear** patterns.

---

### 1.3 Enter: The Neural Network

Think of a neural network as a **team of specialists**:

| Specialist | Focus |
|------------|--------|
| **1** | Luxury indicators — rooms + size combined |
| **2** | Location quality |
| **3** | Budget-friendly markers |
| **4** | Suburban vs urban patterns |

**Final Appraiser:** Combines all opinions → price estimate.

Each "specialist" is a **neuron**. The "team" is a **layer** of neurons.

---

### 1.4 Our Network Architecture

We'll build this structure:

```
INPUT LAYER          HIDDEN LAYER        OUTPUT LAYER
(3 features)         (4 neurons)          (1 prediction)

  [rooms]    \      [Neuron 1]
  [area]      →→→→  [Neuron 2]     →→→→  [price]
  [distance] /      [Neuron 3]
                    [Neuron 4]
```

**Why 4 hidden neurons?**

- **Too few:** Can't learn complex patterns  
- **Too many:** Might "memorize" instead of learning patterns  
- **4** is a good middle ground for our simple dataset  

---

## Part 2: The Mathematics

### 2.1 What Does ONE Neuron Do?

A neuron performs two simple operations.

**Step 1: Weighted sum (linear combination)**

```
z = (w₁ × input₁) + (w₂ × input₂) + (w₃ × input₃) + bias
```

Think of weights as *"how much do I care about this input?"*

**Example:**

```
weights = [w₁ = 0.5, w₂ = 0.3, w₃ = -0.2]   ← rooms & area important, distance slightly negative
inputs  = [rooms = 3, area = 1500, market_distance = 2]
bias    = 0.1

z = (0.5 × 3) + (0.3 × 1500) + (-0.2 × 2) + 0.1
z = 1.5 + 450 + (-0.4) + 0.1 = 451.2
```

**Step 2: Activation function**

```
output = activation(z)
```

This adds **non-linearity**. Without it, stacking many neurons would just create another linear function.

---

### 2.2 The Activation Function: ReLU

We use **ReLU** (Rectified Linear Unit):

```
ReLU(x) = x  if x > 0,  else  0
```

In plain English: keep positive numbers, zero out negative ones.

**Example:**

```
Input:   -5   -2    0    2    5
Output:   0    0    0    2    5
          ↑    ↑    ↑    ↑    ↑
        killed  killed  kept  kept  kept
```

**Why ReLU?**

- Simple to calculate: just `max(0, x)`  
- Helps the network learn faster  
- Helps avoid the "vanishing gradient" problem  
- Introduces non-linearity so the network can learn curves  

**Why not for the output?** Our output is price—it can be any positive number. We want a direct linear mapping at the end, so the output layer has no activation (or we could use a non-negative linear output).

---

### 2.3 Forward Propagation: Making a Prediction

Trace through the **entire** network with real numbers.

**Given:** A house with 3 rooms, 1500 sqft, 2 km from market  
**Want:** Predicted price  

**Initial (random) weights:**

```
Input → Hidden weights:
  Neuron 1: [0.2,  0.1,  -0.3]  bias:  0.1
  Neuron 2: [-0.1, 0.2,   0.1]  bias: -0.05
  Neuron 3: [0.15, -0.1,  0.2]  bias:  0.2
  Neuron 4: [0.3,  0.05, -0.1]  bias: -0.15

Hidden → Output weights:
  Output: [0.5, 0.3, -0.2, 0.1]  bias: 0.05
```

**Hidden layer calculation:**

- **Neuron 1:**  
  `z₁ = (0.2×3) + (0.1×1500) + (-0.3×2) + 0.1 = 150.1`  
  `a₁ = ReLU(150.1) = 150.1` ✓  

- **Neuron 2:**  
  `z₂ = (-0.1×3) + (0.2×1500) + (0.1×2) + (-0.05) = 299.85`  
  `a₂ = ReLU(299.85) = 299.85` ✓  

- **Neuron 3:**  
  `z₃ = (0.15×3) + (-0.1×1500) + (0.2×2) + 0.2 = -148.95`  
  `a₃ = ReLU(-148.95) = 0` ✗ (killed!)  

- **Neuron 4:**  
  `z₄ = (0.3×3) + (0.05×1500) + (-0.1×2) + (-0.15) = 75.55`  
  `a₄ = ReLU(75.55) = 75.55` ✓  

**Hidden layer output:** `[150.1, 299.85, 0, 75.55]`

**Output layer:**

```
z_out = (0.5×150.1) + (0.3×299.85) + (-0.2×0) + (0.1×75.55) + 0.05
z_out = 75.05 + 89.955 + 0 + 7.555 + 0.05 = 172.61
output = 172.61  (no activation for output)
```

**Prediction:** $172,610  

If the actual price is $300,000, this is way off. The weights are random, so predictions are bad—that’s what **learning** fixes.

---

### 2.4 The Problem: Different Scales

Area (1500) is much larger than rooms (3). We had to use small weights (e.g. 0.1) for area so it didn’t dominate. That makes learning harder and unstable.

**Solution:** Normalize all inputs to the same scale, e.g. [0, 1].

**Formula:**

```
normalized = (value - min) / (max - min)
```

**Example:**

- **Rooms:** min=2, max=6 → `3 → (3-2)/(6-2) = 0.25`  
- **Area:** min=900, max=3000 → `1500 → (1500-900)/(3000-900) ≈ 0.286`  
- **Distance:** min=0.8, max=4 → `2.0 → (2.0-0.8)/(4-0.8) = 0.375`  

Inputs become `[0.25, 0.286, 0.375]`—all on the same scale.

---

### 2.5 Measuring Mistakes: Loss Function

We need to measure how wrong the prediction is.

**Mean Squared Error (MSE):**

```
Loss = (predicted - actual)²
```

**Example:** Predicted $172,610, Actual $300,000  
Error = -$127,390 → Loss = (-127,390)² = 16,228,214,521  

**Why square?**

- Makes errors positive  
- Large errors are penalized more than small ones  
- Convenient for gradient-based learning  

---

### 2.6 Learning: Backward Propagation

**Idea:** See how much each weight contributed to the error, then adjust it.

**Hill analogy:** You’re on a foggy hill and want to reach the bottom. You feel which way is downhill and take a step. In our case:

- The **hill** is the error (loss)  
- Your **position** is the current weights  
- **Downhill** is the direction that reduces error  
- A **step** is updating the weights  

The **gradient** tells us the downhill direction for each weight.

**Update rule:**

```
new_weight = old_weight - (learning_rate × gradient)
```

**Learning rate:**

- Too large: overshoot, unstable  
- Too small: very slow  
- Just right: steady progress  

---

### 2.7 Calculating Gradients (Chain Rule)

For the **output** neuron (MSE, linear output):

```
∂Loss/∂output = 2 × (predicted - actual)
∂output/∂weight = hidden_activation
→ gradient = 2 × (predicted - actual) × hidden_activation
```

We often use `(predicted - actual) × hidden_activation` (absorb the 2 into the learning rate).

**Example:** Prediction too low → error negative → gradient for a positive hidden activation is negative → `new_weight = old_weight - (lr × negative) = old_weight + something` → weight increases, so next prediction is higher. ✓  

For **hidden** layers we propagate error backward:

```
hidden_error = output_error × weight × ReLU_derivative
```

**ReLU derivative:**

- **1** if the neuron was active (z > 0)  
- **0** if it was killed (z ≤ 0)  

If a neuron was killed by ReLU, it didn’t contribute to the output, so we don’t update its weights much.

---

## Part 3: Implementation in Python (No External Libraries)

The following implementation uses only the Python standard library (`random`). No PyTorch, NumPy, or other packages.

```python
"""
Neural network for house price prediction.
Architecture: 3 inputs → 4 hidden (ReLU) → 1 output (linear).
Uses only Python standard library (random).
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

    nn = NeuralNetwork(seed=42)
    learning_rate = 0.1
    epochs = 500

    print("Training (MSE on normalized target)...")
    for epoch in range(epochs):
        total_loss = 0.0
        for features, price in data:
            inputs_norm, target_norm = prepare_sample(features, price)
            prediction, hidden_pre_activations, hidden_activations = nn.forward(inputs_norm)
            loss = (prediction - target_norm) ** 2
            total_loss += loss
            nn.backward(inputs_norm, hidden_pre_activations, hidden_activations, prediction, target_norm, learning_rate)
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}, mean loss: {total_loss / len(data):.6f}")

    # Predict on first house and denormalize
    inputs_test, _ = prepare_sample([3, 1500, 2.0], 300_000)
    prediction_norm, _, _ = nn.forward(inputs_test)
    pred_price = denormalize(prediction_norm, price_min, price_max)
    print(f"\nExample: [3 rooms, 1500 sqft, 2 km] -> predicted ${pred_price:,.0f} (actual $300,000)")

    return nn


if __name__ == "__main__":
    train_and_predict()
```

---

### How to run

Save the code block above as e.g. `neural_network_house_price.py` in the same directory as your notes, or copy it into a single `.py` file. Run:

```bash
python neural_network_house_price.py
```

You should see the mean loss decrease over epochs and a sample prediction for the house `[3, 1500, 2.0]`. With this tiny dataset and no regularization, the model will tend to overfit; for real use you’d add more data, validation, and possibly more epochs or a smaller learning rate.

This matches the math in Part 2: normalized inputs, one hidden layer with ReLU, linear output, MSE, and gradient updates via the chain rule and ReLU derivative.
