"""
Track 3.1: Neural Networks - Super Simple Edition
=================================================
Build your first neural network from scratch with clear explanations.

Author: AI Engineering Masterclass
"""

print("=" * 60)
print("  WHAT IS A NEURAL NETWORK?")
print("=" * 60)

print("""
[BRAIN] THE HUMAN BRAIN ANALOGY:

   Your brain has BILLIONS of tiny cells called neurons
   Each neuron receives signals, processes them, and sends output
   Neurons are connected in a network

[BOT] ARTIFICIAL NEURAL NETWORK:

   We mimic this with math!
   - Input neurons: receive data
   - Hidden neurons: process data
   - Output neurons: give predictions

   Think of it like a TEAM OF VOTERS:
   - Each person (neuron) makes a decision
   - They all vote
   - The final answer is the majority vote
""")

# ==============================================================================
# PART 1: A SINGLE NEURON
# ==============================================================================

print("\n" + "=" * 60)
print("  PART 1: A SINGLE NEURON")
print("=" * 60)

print("""
   [NUM] A NEURON DOES 3 THINGS:

   1. WEIGHT: Multiply input by weight
      y = x * w

   2. ADD: Sum up all weighted inputs
      y = x1*w1 + x2*w2 + x3*w3 + ...

   3. ACTIVATION: Decide if output "fires"
      if sum > threshold: fire! (output = 1)
      else: don't fire (output = 0)
""")

def simple_neuron():
    """A single neuron calculation."""
    # Input
    x = 0.8  # Some input value

    # Weight (how important is this input?)
    w = 0.5  # Weight

    # Step 1 & 2: Weighted sum
    weighted_sum = x * w

    # Step 3: Activation (ReLU - simpler than step function)
    # ReLU: if positive, keep it; if negative, make it 0
    if weighted_sum > 0:
        output = weighted_sum
    else:
        output = 0

    print(f"   Input: {x}")
    print(f"   Weight: {w}")
    print(f"   Weighted sum: {weighted_sum}")
    print(f"   After ReLU activation: {output}")

print("\n[CALC] Let's run a simple neuron:")
simple_neuron()

# ==============================================================================
# PART 2: LAYERS OF NEURONS
# ==============================================================================

print("\n" + "=" * 60)
print("  PART 2: LAYERS OF NEURONS")
print("=" * 60)

print("""
   [CHART] A LAYER = Multiple neurons working together

   Input (1 value) -> Layer of 4 neurons -> Output (4 values)

   Example: Processing an image
   - Input: pixel value (0.5)
   - Hidden layer: 4 neurons, each with different weights
   - Each neuron learns something different (edges, colors, etc.)
""")

import numpy as np

def layer_forward():
    """Forward pass through one layer."""
    # Input (3 values)
    inputs = np.array([0.5, 0.3, 0.8])

    # Weights for 4 neurons (each neuron has 3 weights)
    weights = np.array([
        [0.1, 0.4, 0.2],  # Neuron 1
        [0.3, 0.1, 0.5],  # Neuron 2
        [0.2, 0.5, 0.1],  # Neuron 3
        [0.4, 0.2, 0.3],  # Neuron 4
    ])

    # Step 1: Multiply input by weights for each neuron
    # matrix multiplication: (4×3) @ (3,) = (4,)
    outputs = np.dot(weights, inputs)

    # Step 2: Apply ReLU activation
    outputs = np.maximum(0, outputs)

    print(f"   Input: {inputs}")
    print(f"   Weights shape: {weights.shape} (4 neurons, 3 inputs each)")
    print(f"   Layer outputs: {outputs.round(4)}")

print("\n[RUN] Forward pass through a layer:")
layer_forward()

# ==============================================================================
# PART 3: A COMPLETE NETWORK
# ==============================================================================

print("\n" + "=" * 60)
print("  PART 3: COMPLETE NEURAL NETWORK")
print("=" * 60)

print("""
   [BUILD] FULL NETWORK STRUCTURE:

   +-------------------------------------+
   |  INPUT LAYER (3 neurons)            |
   |  [x1] [x2] [x3]                     |
   +-------------+----------------------+
                 |
                 v
   +-------------------------------------+
   |  HIDDEN LAYER 1 (4 neurons)          |
   |  [h1] [h2] [h3] [h4]                |
   +-------------+----------------------+
                 |
                 v
   +-------------------------------------+
   |  HIDDEN LAYER 2 (4 neurons)          |
   |  [h5] [h6] [h7] [h8]                |
   +-------------+----------------------+
                 |
                 v
   +-------------------------------------+
   |  OUTPUT LAYER (1 neuron)             |
   |  [y_hat]                             |
   +-------------------------------------+
""")

def simple_network():
    """A complete forward pass."""
    # Layer dimensions
    input_size = 3
    hidden1_size = 4
    hidden2_size = 4
    output_size = 1

    # Random weights for demonstration
    np.random.seed(42)

    W1 = np.random.randn(input_size, hidden1_size) * 0.5
    b1 = np.zeros(hidden1_size)
    W2 = np.random.randn(hidden1_size, hidden2_size) * 0.5
    b2 = np.zeros(hidden2_size)
    W3 = np.random.randn(hidden2_size, output_size) * 0.5
    b3 = np.zeros(output_size)

    # Input: some data
    x = np.array([0.5, 0.3, 0.8])

    print(f"   Input: {x}")

    # Layer 1
    h1 = np.dot(x, W1) + b1
    h1 = np.maximum(0, h1)  # ReLU
    print(f"   After Layer 1 (ReLU): {h1.round(4)}")

    # Layer 2
    h2 = np.dot(h1, W2) + b2
    h2 = np.maximum(0, h2)  # ReLU
    print(f"   After Layer 2 (ReLU): {h2.round(4)}")

    # Output
    y = np.dot(h2, W3) + b3
    print(f"   Final output: {y.round(4)}")

print("\n[RUN] Full network forward pass:")
simple_network()

# ==============================================================================
# PART 4: TRAINING (GRADIENT DESCENT)
# ==============================================================================

print("\n" + "=" * 60)
print("  PART 4: HOW NETWORKS LEARN")
print("=" * 60)

print("""
   [BOOK] TRAINING = "Guess, Check, Improve" Loop

   1. FORWARD: Make a prediction
      Input -> Layer 1 -> Layer 2 -> Output

   2. COMPARE: How wrong was I?
      Loss = (prediction - actual)^2

   3. BACKWARD: How should I adjust?
      Calculate gradients (which direction to move)

   4. UPDATE: Make small adjustments
      new_weight = old_weight - learning_rate * gradient

   5. REPEAT: Do this thousands of times
      -> Weights get better -> Predictions get better!
""")

def gradient_descent_demo():
    """Simple gradient descent demonstration."""
    print("\n[TARGET] Goal: Find the minimum of y = (x-3)^2")

    # Start with random x
    x = 0.0
    learning_rate = 0.1

    print(f"\n   Starting position: x = {x}")

    for step in range(1, 6):
        # Gradient of (x-3)² is 2(x-3)
        gradient = 2 * (x - 3)

        # Update rule: x = x - learning_rate * gradient
        x_new = x - learning_rate * gradient

        # Calculate current loss
        loss = (x - 3) ** 2

        print(f"   Step {step}: x = {x:.4f}, loss = {loss:.4f}, gradient = {gradient:.4f}")

        x = x_new

    print(f"\n   Final: x = {x:.4f} (close to target x=3!)")

print("\n[LOOK] Watch gradient descent find the minimum:")
gradient_descent_demo()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
[OK] Neural Network = Layers of artificial neurons

[OK] Forward Pass: Input -> Process -> Output

[OK] Activation: ReLU turns negatives to 0 (keeps positives)

[OK] Training Loop:
   1. Forward pass (make prediction)
   2. Calculate loss (how wrong)
   3. Backward pass (find gradients)
   4. Update weights (reduce loss)
   5. Repeat!

[OK] More layers = Can learn more complex patterns

NEXT: Build this in PyTorch (real neural network library)!
""")