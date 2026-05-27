"""
Track 3.1: Neural Networks From Scratch
========================================
Build a neural network from scratch using only NumPy:
- Forward pass
- Backpropagation
- Training loop
- Visualization

Author: AI Engineering Masterclass
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, clear_output
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==============================================================================
# PART 1: NEURAL NETWORK ARCHITECTURE
# ==============================================================================

class NeuralNetwork:
    """
    A simple fully-connected neural network built from scratch.

    Architecture:
    - Input layer: 2 neurons
    - Hidden layer 1: 4 neurons (ReLU activation)
    - Hidden layer 2: 4 neurons (ReLU activation)
    - Output layer: 1 neuron (Sigmoid activation)

    Forward pass: Input → Linear → ReLU → Linear → ReLU → Linear → Sigmoid → Output
    """

    def __init__(self, layer_sizes):
        """
        Initialize network with given layer sizes.

        Args:
            layer_sizes: List of integers defining each layer's size
                        e.g., [2, 4, 4, 1] for 2-input, 2 hidden layers (4 each), 1 output
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier initialization for stable training
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

        # For visualization
        self.activations = []
        self.z_values = []  # Pre-activation values

    def sigmoid(self, z):
        """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        """Derivative: σ'(z) = σ(z) * (1 - σ(z))"""
        return a * (1 - a)

    def relu(self, z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU derivative: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

    def forward(self, X):
        """
        Forward pass through the network.

        For each layer:
        1. Compute z = X·W + b (linear transformation)
        2. Apply activation a = f(z)

        Returns:
            Output prediction and cache of intermediate values
        """
        self.activations = [X]  # Store input
        self.z_values = []

        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            # Apply activation (except for last layer if using softmax)
            if i < self.num_layers - 2:
                a = self.relu(z)
            else:
                # Output layer: sigmoid for binary classification
                a = self.sigmoid(z)

            self.activations.append(a)

        return self.activations[-1]

    def backward(self, y, learning_rate=0.1):
        """
        Backpropagation algorithm.

        For each layer (reverse order):
        1. Compute gradient of loss w.r.t. weights/biases
        2. Propagate gradient to previous layer

        The chain rule is the foundation:
        ∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W
        """
        m = y.shape[0]  # Number of samples
        num_layers = len(self.weights)

        # Output layer gradient
        # For BCE loss: L = -(y·log(ŷ) + (1-y)·log(1-ŷ))
        # ∂L/∂ŷ = -(y/ŷ) + (1-y)/(1-ŷ)
        delta = self.activations[-1] - y.reshape(-1, 1)

        # Store gradients
        weight_gradients = []
        bias_gradients = []

        for i in reversed(range(num_layers)):
            # Compute gradients
            grad_W = np.dot(self.activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            weight_gradients.insert(0, grad_W)
            bias_gradients.insert(0, grad_b)

            # Backpropagate delta to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.relu_derivative(self.z_values[i - 1])

        # Update weights using gradient descent
        for i in range(num_layers):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        m = y_true.shape[0]
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.sum(loss) / m

    def predict(self, X):
        """Make predictions (forward pass only)."""
        output = self.forward(X)
        return (output >= 0.5).astype(int).flatten()

    def accuracy(self, y_true, y_pred):
        """Compute accuracy."""
        return np.mean(y_true == y_pred)

# ==============================================================================
# PART 2: TRAINING LOOP
# ==============================================================================

def generate_data():
    """Generate XOR dataset - requires non-linear decision boundary."""
    # XOR pattern - cannot be separated by a linear model
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0.2, 0.2], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2],
        [0.3, 0.3], [0.7, 0.7], [0.3, 0.7], [0.7, 0.3],
        [0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1],
        [0.4, 0.4], [0.6, 0.6], [0.4, 0.6], [0.6, 0.4],
        [0.15, 0.15], [0.85, 0.85], [0.15, 0.85], [0.85, 0.15],
    ])
    y = np.array([0, 1, 1, 0] * 6)  # XOR pattern
    return X, y

def train_network(nn, X, y, epochs=10000, learning_rate=0.1, verbose=True):
    """Train the neural network."""

    losses = []
    accuracies = []

    for epoch in range(epochs):
        # Forward pass
        output = nn.forward(X)

        # Compute loss (for logging)
        loss = nn.compute_loss(y, output)
        predictions = nn.predict(X)
        accuracy = nn.accuracy(y, predictions)

        losses.append(loss)
        accuracies.append(accuracy)

        # Backward pass (gradient descent)
        nn.backward(y, learning_rate)

        if verbose and (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch + 1:5d} | Loss: {loss:.6f} | Accuracy: {accuracy:.4f}")

    return losses, accuracies

def visualize_training(losses, accuracies):
    """Visualize training progress."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(accuracies, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy Over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("Saved: training_curves.png")

def visualize_decision_boundary(nn, X, y):
    """Visualize the learned decision boundary."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Neural network prediction
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax1 = axes[0]
    ax1.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn, alpha=0.6)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlGn,
                edgecolors='black', s=100, linewidths=1.5)
    ax1.set_xlabel('Input X₁', fontsize=12)
    ax1.set_ylabel('Input X₂', fontsize=12)
    ax1.set_title('Learned Decision Boundary', fontsize=14)

    # Plot 2: Visualize network architecture
    ax2 = axes[1]

    layer_sizes = [2, 4, 4, 1]
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']

    max_neurons = max(layer_sizes)
    y_positions = np.linspace(0.9, 0.1, max_neurons + 1)[1:]
    x_positions = np.linspace(0.15, 0.85, len(layer_sizes))

    # Draw connections (simplified)
    for l in range(len(layer_sizes) - 1):
        n_curr = layer_sizes[l]
        n_next = layer_sizes[l + 1]

        y_curr = np.linspace(0.5, 0.5, n_curr) if n_curr <= 3 else y_positions[:n_curr]
        y_next = np.linspace(0.5, 0.5, n_next) if n_next <= 3 else y_positions[:n_next]

        # Only show a few connections to avoid clutter
        step = max(1, n_curr // 3)
        for i in range(0, n_curr, step):
            for j in range(0, n_next, step):
                ax2.plot([x_positions[l], x_positions[l + 1]],
                         [y_curr[min(i, len(y_curr) - 1)],
                          y_next[min(j, len(y_next) - 1)]],
                         'gray', alpha=0.2, linewidth=0.5)

        # Draw neurons
        for i in range(n_curr):
            color = 'lightblue' if l > 0 else 'lightgreen'
            ax2.scatter([x_positions[l]], [y_curr[i]], s=400,
                       c=color, edgecolors='black', zorder=5)

        for i in range(n_next):
            color = 'lightcoral' if l == len(layer_sizes) - 2 else 'lightyellow'
            ax2.scatter([x_positions[l + 1]], [y_next[i]], s=400,
                       c=color, edgecolors='black', zorder=5)

        ax2.text(x_positions[l], 0.98, layer_names[l], ha='center', fontsize=10, fontweight='bold')
        ax2.text(x_positions[l], 0.93, f'({layer_sizes[l]} neurons)', ha='center', fontsize=8)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Network Architecture', fontsize=14)

    plt.tight_layout()
    plt.savefig('decision_boundary.png', dpi=150)
    plt.show()
    print("Saved: decision_boundary.png")

# ==============================================================================
# PART 3: FORWARD & BACKWARD PASS VISUALIZATION
# ==============================================================================

def visualize_forward_pass():
    """Step-by-step forward pass visualization."""

    print("\n" + "=" * 70)
    print("  FORWARD PASS VISUALIZATION")
    print("=" * 70)

    # Simple 2-2-1 network with known weights
    nn = NeuralNetwork([2, 2, 1])

    # Set deterministic weights for demonstration
    nn.weights[0] = np.array([[0.5, 0.3], [-0.2, 0.4]])
    nn.weights[1] = np.array([[0.7], [-0.3]])
    nn.biases[0] = np.array([[0.1, -0.1]])
    nn.biases[1] = np.array([[0.0]])

    # Input
    x = np.array([[0.5, 0.8]])

    print("\nInput:", x)
    print("\nLayer 0 (Input):", x)

    # Layer 1
    z1 = np.dot(x, nn.weights[0]) + nn.biases[0]
    a1 = nn.relu(z1)
    print("\nLayer 1:")
    print(f"  Pre-activation (z₁):\n  {z1}")
    print(f"  After ReLU (a₁):\n  {a1}")

    # Layer 2 (Output)
    z2 = np.dot(a1, nn.weights[1]) + nn.biases[1]
    a2 = nn.sigmoid(z2)
    print("\nLayer 2 (Output):")
    print(f"  Pre-activation (z₂):\n  {z2}")
    print(f"  After Sigmoid (ŷ):\n  {a2}")

    print("\n" + "-" * 50)
    print("BACKWARD PASS (Gradient Flow)")
    print("-" * 50)

    # Target
    y = np.array([1])

    # Output gradient
    delta_output = a2 - y.reshape(-1, 1)
    print(f"\nOutput error (ŷ - y):\n  {delta_output}")

    # Layer 2 gradients
    grad_W1 = np.dot(a1.T, delta_output)
    gap_b1 = np.sum(delta_output, axis=0, keepdims=True)
    print(f"\nWeight gradients dL/dW₁:\n  {grad_W1}")

    # Backprop into hidden layer
    delta_hidden = np.dot(delta_output, nn.weights[1].T)
    delta_hidden *= nn.relu_derivative(z1)
    print(f"\nDelta propagated to hidden: {delta_hidden}")

    # Layer 1 weight gradients
    grad_W0 = np.dot(x.T, delta_hidden)
    print(f"\nWeight gradients dL/dW₀:\n  {grad_W0}")

def visualize_gradient_flow():
    """Visualize vanishing gradient problem."""

    print("\n" + "=" * 70)
    print("  VANISHING GRADIENT DEMONSTRATION")
    print("=" * 70)

    # Create networks of different depths
    depths = [2, 3, 4, 5]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, depth in enumerate(depths):
        ax = axes[idx]

        # Simple network with small activations
        hidden_size = 4
        layer_sizes = [1] + [hidden_size] * depth + [1]

        nn = NeuralNetwork(layer_sizes)

        # Run forward pass
        x = np.array([[1.0]])
        y = np.array([[1]])

        nn.forward(x)

        # Manually compute gradients at each layer
        gradient_magnitudes = []

        delta = nn.activations[-1] - y.reshape(-1, 1)

        for i in reversed(range(len(nn.weights))):
            grad = np.sum(np.abs(delta))
            gradient_magnitudes.append(grad)

            if i > 0:
                delta = np.dot(delta, nn.weights[i].T)
                delta *= nn.relu_derivative(nn.z_values[i - 1])

        gradient_magnitudes = gradient_magnitudes[::-1]

        # Plot gradient magnitudes
        axes[idx].bar(range(len(gradient_magnitudes)), gradient_magnitudes,
                      color=plt.cm.Blues(np.linspace(0.3, 0.9, len(gradient_magnitudes))))
        axes[idx].set_xlabel('Layer')
        axes[idx].set_ylabel('|Gradient|')
        axes[idx].set_title(f'{depth} Hidden Layers')
        axes[idx].set_yscale('log')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Gradient Magnitude Through Layers\n(Vanishing Gradient Problem)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=150)
    plt.show()
    print("Saved: gradient_flow.png")

    print("\nNote: Gradients shrink exponentially as they propagate back.")
    print("This is why deeper networks are harder to train.")
    print("Solution: Skip connections (ResNet), LSTM/GRU gates, careful initialization")

# ==============================================================================
# PART 4: ACTIVATION FUNCTION COMPARISON
# ==============================================================================

def compare_activations():
    """Compare different activation functions."""

    print("\n" + "=" * 70)
    print("  ACTIVATION FUNCTION COMPARISON")
    print("=" * 70)

    z = np.linspace(-5, 5, 100)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-z))

    # Tanh
    tanh = np.tanh(z)

    # ReLU
    relu = np.maximum(0, z)

    # Leaky ReLU
    leakyrelu = np.where(z > 0, z, 0.01 * z)

    # ELU
    elu = np.where(z > 0, z, np.exp(z) - 1)

    # GELU (approximation)
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    gelu_vals = gelu(z)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    activations = [
        ('Sigmoid', sigmoid, 'σ(z) = 1/(1+e^(-z))', 'Maps to (0, 1)'),
        ('Tanh', tanh, 'tanh(z)', 'Maps to (-1, 1)'),
        ('ReLU', relu, 'max(0, z)', 'Simple, sparse'),
        ('Leaky ReLU', leakyrelu, 'max(0.01z, z)', 'No dying ReLU'),
        ('ELU', elu, 'z if z>0, e^z-1 if z≤0', 'Smooth, negative vals'),
        ('GELU', gelu_vals, '0.5z(1+tanh(...))', 'Used in Transformers')
    ]

    for idx, (name, vals, formula, note) in enumerate(activations):
        ax = axes[idx // 3, idx % 3]
        ax.plot(z, vals, 'b-', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-1.5, 2)
        ax.set_title(f'{name}\n{formula}\n({note})', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Activation Functions Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('activations.png', dpi=150)
    plt.show()
    print("Saved: activations.png")

    print("\nActivation Function Properties:")
    print("-" * 50)
    print("Sigmoid:  Range (0,1), gradient vanishes for |z|>>0")
    print("Tanh:     Range (-1,1), zero-centered, gradient vanishes")
    print("ReLU:     Range [0,∞), efficient, but 'dying ReLU' problem")
    print("LeakyReLU: Range (-∞,∞), allows small negative gradient")
    print("ELU:      Smooth everywhere, but slightly slower")
    print("GELU:     Best for Transformers, used in BERT, GPT, etc.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TRACK 3.1: NEURAL NETWORKS FROM SCRATCH")
    print("=" * 70)

    # Generate XOR data
    print("\nGenerating XOR dataset...")
    X, y = generate_data()
    print(f"Data shape: {X.shape}, Labels: {y.shape}")

    # Create and train network
    print("\nCreating neural network: [2, 4, 4, 1]")
    nn = NeuralNetwork([2, 4, 4, 1])

    print("\nTraining network...")
    losses, accuracies = train_network(nn, X, y, epochs=10000, learning_rate=0.1)

    # Visualize results
    visualize_training(losses, accuracies)
    visualize_decision_boundary(nn, X, y)

    # Detailed forward/backward pass
    visualize_forward_pass()
    visualize_gradient_flow()

    # Activation comparison
    compare_activations()

    # Final evaluation
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    predictions = nn.predict(X)
    print(f"\nFinal Accuracy: {nn.accuracy(y, predictions):.2%}")
    print("\nPredictions vs Actual:")
    print("Input    | Actual | Predicted | Correct?")
    print("-" * 45)
    for i in range(len(X)):
        pred = '1' if predictions[i] else '0'
        actual = '1' if y[i] else '0'
        correct = '✓' if predictions[i] == y[i] else '✗'
        print(f"({X[i][0]:.1f}, {X[i][1]:.1f})  |    {actual}    |     {pred}      |   {correct}")

    print("\n" + "=" * 70)
    print("  EXERCISES FOR THE READER")
    print("=" * 70)
    print("""
    1. Modify the network:
       - Change layer sizes: try [2, 8, 8, 4, 1]
       - Add more layers
       - Change activation functions

    2. Experiment with learning:
       - Compare learning rates: 0.01 vs 0.1 vs 1.0
       - Implement momentum updates
       - Implement learning rate decay

    3. Try different datasets:
       - Classify a circle (inner vs outer ring)
       - Spiral pattern
       - Your own dataset

    4. Implement from scratch:
       - Mini-batch gradient descent
       - Momentum optimizer
       - Adam optimizer
    """)

    print("\n" + "=" * 70)
    print("  NEXT: CNNs, RNNs, and PyTorch Introduction")
    print("=" * 70)
