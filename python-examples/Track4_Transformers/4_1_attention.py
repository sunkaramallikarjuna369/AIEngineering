"""
Track 4.1: Attention Mechanism Implementation
============================================
Build attention from scratch to understand how LLMs work.

Key concepts:
- Scaled dot-product attention
- Query, Key, Value projections
- Multi-head attention
- Causal masking (for autoregressive models)

Author: AI Engineering Masterclass
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==============================================================================
# PART 1: ATTENTION INTUITION
# ==============================================================================

def attention_intuition():
    """
    Attention allows each token to "look at" all other tokens
    and weighted-sum their information.

    Think of it like a dictionary catalog:
    - Query: What information am I looking for?
    - Key: What information do I contain?
    - Value: What actual information to retrieve?

    The attention score measures how well Query matches Key.
    """

    print("=" * 70)
    print("  ATTENTION MECHANISM INTUITION")
    print("=" * 70)

    print("""
    Classic analogy: Library Search

    📚 Library catalog analogy:

    KEYS = Index cards in catalog drawers (what each drawer contains)
    QUERIES = Your search terms (what you're looking for)
    VALUES = Actual books/documents (the actual information)

    When you search a library:
    1. Your query is matched against all index cards
    2. Best matches get higher weights
    3. You retrieve documents from highest-weighted drawers

    Attention does the SAME thing mathematically!
    - Q·K determines match quality
    - Softmax creates valid probability distribution
    - Weighted sum of V retrieves actual content
    """)

def scaled_dot_product_attention():
    """
    Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(QK^T / √d) × V

    Steps:
    1. Compute similarity scores: QK^T
    2. Scale by √d (prevent vanishing gradients)
    3. Softmax to get attention weights
    4. Weighted sum of values
    """

    print("\n" + "-" * 50)
    print("SCALED DOT-PRODUCT ATTENTIONMathematics")
    print("-" * 50)

    # Simple example: "The cat sat on the mat"
    sentence = ["The", "cat", "sat", "on", "the", "mat"]

    # Create simple one-hot embeddings for demo
    vocab_size = 6
    d_model = 4  # Embedding dimension

    # Simple embeddings (normally learned)
    embeddings = {
        "The": np.array([1, 0, 0, 0]),
        "cat": np.array([0, 1, 0, 0]),
        "sat": np.array([0, 0, 1, 0]),
        "on":  np.array([1, 1, 0, 0]),
        "the": np.array([1, 0, 1, 0]),
        "mat": np.array([0, 1, 1, 0]),
    }

    X = np.array([embeddings[w] for w in sentence]).astype(float)

    print(f"Input sequence: {sentence}")
    print(f"Shape: {X.shape} (seq_len={X.shape[0]}, d_model={X.shape[1]})")

    # Simple Q, K, V projections (identity for simplicity)
    d_k = d_model  # Often d_k = d_model / num_heads

    Q = X  # Query = input (simplified)
    K = X  # Key = input
    V = X  # Value = input

    print(f"\nQ, K, V shapes: {Q.shape}")

    # Step 1: Compute attention scores
    scores = np.dot(Q, K.T)
    print(f"\n1. Attention scores (QK^T):\n{scores.round(2)}")

    # Step 2: Scale
    scaled_scores = scores / np.sqrt(d_k)
    print(f"\n2. Scaled scores (÷ √d):\n{scaled_scores.round(2)}")

    # Step 3: Softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_weights = softmax(scaled_scores)
    print(f"\n3. Attention weights (softmax):\n{attention_weights.round(2)}")

    # Step 4: Weighted sum
    output = np.dot(attention_weights, V)
    print(f"\n4. Output (weighted sum of V):\n{output.round(2)}")

    return attention_weights

def visualize_attention():
    """Visualize attention patterns."""

    print("\n" + "-" * 50)
    print("ATTENTION PATTERN VISUALIZATION")
    print("-" * 50)

    # "The cat sat on the mat" - expected: "cat" attends to "The" and itself
    # "sat" attends to its subject and object
    sentence = ["The", "cat", "sat", "on", "the", "mat"]

    # Simulated attention pattern (more realistic)
    # In practice, this is learned
    attention_pattern = np.array([
        [0.5,  0.3,  0.05, 0.05, 0.05, 0.05],  # The
        [0.3,  0.5,  0.1,  0.05, 0.05, 0.05],  # cat
        [0.05, 0.3,  0.3,  0.15, 0.1,  0.1 ],  # sat
        [0.05, 0.05, 0.15, 0.4,  0.3,  0.05],  # on
        [0.05, 0.05, 0.1,  0.05, 0.6,  0.15],  # the
        [0.05, 0.05, 0.1,  0.1,  0.1,  0.6 ],  # mat
    ])

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_pattern, cmap='Blues')

    # Add labels
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))
    ax.set_xticklabels(sentence, fontsize=12)
    ax.set_yticklabels(sentence, fontsize=12)

    # Add values to cells
    for i in range(len(sentence)):
        for j in range(len(sentence)):
            text = ax.text(j, i, f'{attention_pattern[i, j]:.2f}',
                          ha='center', va='center', fontsize=9,
                          color='white' if attention_pattern[i, j] > 0.3 else 'black')

    ax.set_xlabel('Key (attended to)', fontsize=12)
    ax.set_ylabel('Query (attending)', fontsize=12)
    ax.set_title('Attention Weights: "The cat sat on the mat"', fontsize=14)

    plt.colorbar(im, label='Attention Weight')
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150)
    plt.show()
    print("Saved: attention_weights.png")

    print("\nInterpretation:")
    print("- Row = Query (who is looking)")
    print("- Column = Key (who is being looked at)")
    print("- Cell value = Attention weight")
    print("\nExample: 'cat' row shows it attends most to itself (0.5) and 'The' (0.3)")

# ==============================================================================
# PART 2: MULTI-HEAD ATTENTION
# ==============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention:
    - Runs attention in parallel multiple times
    - Each "head" learns different aspects of relationships
    - Outputs are concatenated and projected

    Why multiple heads?
    - Different heads can focus on different relationships:
      Head 1: Subject-verb agreements
      Head 2: Proximity in sentence
      Head 3: Semantic similarity
      ...
    """

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Weight matrices (initialized randomly)
        self.W_Q = [np.random.randn(d_model, self.d_k) * 0.1
                   for _ in range(num_heads)]
        self.W_K = [np.random.randn(d_model, self.d_k) * 0.1
                   for _ in range(num_heads)]
        self.W_V = [np.random.randn(d_model, self.d_k) * 0.1
                   for _ in range(num_heads)]
        self.W_O = np.random.randn(d_model, d_model) * 0.1

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass through multi-head attention.

        Args:
            Q: Query tensor (batch, seq_len, d_model)
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = Q.shape
        outputs = []

        for head in range(self.num_heads):
            # Project to d_k
            Q_h = np.dot(Q, self.W_Q[head])
            K_h = np.dot(K, self.W_K[head])
            V_h = np.dot(V, self.W_V[head])

            # Scaled dot-product attention for this head
            scores_h = np.dot(Q_h, K_h.transpose(0, 2, 1)) / np.sqrt(self.d_k)

            # Apply mask if provided
            if mask is not None:
                scores_h = np.where(mask, scores_h, -1e9)

            # Softmax
            exp_scores = np.exp(scores_h - np.max(scores_h, axis=-1, keepdims=True))
            attn_weights_h = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Weighted sum
            head_output = np.dot(attn_weights_h, V_h)
            outputs.append(head_output)

        # Concatenate heads: (batch, seq_len, num_heads * d_k)
        concat_output = np.concatenate(outputs, axis=-1)

        # Final linear projection
        output = np.dot(concat_output, self.W_O)

        return output, outputs  # Return both final and per-head for analysis

def multi_head_demo():
    """Demonstrate multi-head attention."""

    print("\n" + "-" * 50)
    print("MULTI-HEAD ATTENTION DEMO")
    print("-" * 50)

    # Create simple input (batch=1, seq=4, d_model=8)
    seq_len, d_model, num_heads = 4, 8, 2

    X = np.random.randn(1, seq_len, d_model)

    print(f"Input shape: {X.shape}")
    print(f"Hidden dimension: {d_model}")
    print(f"Number of heads: {num_heads}")
    print(f"Dimension per head: {d_model // num_heads}")

    # Create attention
    mha = MultiHeadAttention(d_model, num_heads)

    # Forward pass
    output, head_outputs = mha.forward(X, X, X)

    print(f"\nOutput shape: {output.shape}")
    print(f"Number of head outputs: {len(head_outputs)}")
    print(f"Each head output shape: {head_outputs[0].shape}")

    # Visualize multi-head concept
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Conceptual diagram
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)

    # Input
    ax1.add_patch(plt.Rectangle((0.5, 2), 1.5, 2, fill=True, color='lightblue', edgecolor='black'))
    ax1.text(1.25, 3, 'Input\nSeq×D', ha='center', va='center', fontsize=10)
    ax1.text(1.25, 1.5, 'X', ha='center', fontsize=12)

    # Heads
    for i, (x, color) in enumerate([(3.5, 'lightgreen'), (6.5, 'lightyellow')]):
        ax1.add_patch(plt.Rectangle((x, 1.5), 2, 3, fill=True, color=color, edgecolor='black'))
        ax1.text(x+1, 3, f'Head {i+1}', ha='center', va='center', fontsize=10)

    # Output
    ax1.add_patch(plt.Rectangle((9, 2), 1.5, 2, fill=True, color='lightcoral', edgecolor='black'))
    ax1.text(9.75, 3, 'Output\nConcat', ha='center', va='center', fontsize=10)
    ax1.text(9.75, 1.5, 'Y', ha='center', fontsize=12)

    # Arrows
    ax1.annotate('', xy=(3.5, 3), xytext=(2, 3), arrowprops=dict(arrowstyle='->', lw=2))
    ax1.annotate('', xy=(6.5, 3), xytext=(5.5, 3), arrowprops=dict(arrowstyle='->', lw=2))
    ax1.annotate('', xy=(9, 3), xytext=(8.5, 3), arrowprops=dict(arrowstyle='->', lw=2))

    ax1.axis('off')
    ax1.set_title('Multi-Head Attention Architecture', fontsize=12)

    # Per-head attention patterns (simulated)
    x_labels = ['Token\nA', 'Token\nB', 'Token\nC', 'Token\nD']

    for head_idx in range(2):
        ax = axes[head_idx + 1]
        # Simulated patterns for two different aspects
        if head_idx == 0:
            attn = np.array([[0.6, 0.3, 0.1, 0.0],
                           [0.3, 0.6, 0.1, 0.0],
                           [0.1, 0.1, 0.5, 0.3],
                           [0.0, 0.0, 0.3, 0.7]])
            title = 'Head 1: Positional patterns'
        else:
            attn = np.array([[0.4, 0.4, 0.1, 0.1],
                           [0.4, 0.4, 0.1, 0.1],
                           [0.1, 0.1, 0.4, 0.4],
                           [0.1, 0.1, 0.4, 0.4]])
            title = 'Head 2: Semantic patterns'

        im = ax.imshow(attn, cmap='Blues')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_yticklabels(x_labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=150)
    plt.show()
    print("Saved: multi_head_attention.png")

# ==============================================================================
# PART 3: CAUSAL (TRIANGULAR) MASKING
# ==============================================================================

def causal_masking():
    """
    Causal masking ensures position i can only attend to positions 0...i.

    This is ESSENTIAL for autoregressive (decoder-only) models like GPT.
    Prevents seeing future tokens during training.
    """

    print("\n" + "-" * 50)
    print("CAUSAL MASKING (Autoregressive)")
    print("-" * 50)

    seq_len = 5

    # Create causal mask (lower triangular)
    mask = np.tril(np.ones((seq_len, seq_len)))

    print(f"Sequence length: {seq_len}")
    print(f"\nCausal mask (1=attend, 0=masked):\n{mask.astype(int)}")

    # Apply mask to attention scores
    fake_scores = np.random.randn(seq_len, seq_len)
    masked_scores = np.where(mask == 1, fake_scores, -1e9)

    print(f"\nRaw attention scores:\n{fake_scores.round(2)}")
    print(f"\nAfter masking (future positions → -∞):\n{masked_scores.round(2)}")

    # After softmax, masked positions get ~0 weight
    exp_scores = np.exp(fake_scores - np.max(fake_scores, axis=-1, keepdims=True))
    masked_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    # Correct: re-mask after softmax
    masked_weights = np.where(mask == 1, masked_weights, 0)

    print(f"\nAttention weights (masked positions → 0):\n{masked_weights.round(3)}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.ones((seq_len, seq_len)), cmap='Blues')
    axes[0].set_title('Full Attention (for encoder)')
    axes[0].set_xlabel('Key position')
    axes[0].set_ylabel('Query position')

    axes[1].imshow(mask, cmap='Blues')
    axes[1].set_title('Causal Mask (for decoder)')
    axes[1].set_xlabel('Key position')
    axes[1].set_ylabel('Query position')

    axes[2].imshow(masked_weights, cmap='Blues')
    axes[2].set_title('Resulting Weights')
    axes[2].set_xlabel('Key position')
    axes[2].set_ylabel('Query position')

    plt.tight_layout()
    plt.savefig('causal_masking.png', dpi=150)
    plt.show()
    print("Saved: causal_masking.png")

# ==============================================================================
# PART 4: TRANSFORMER ATTENTION LAYER
# ==============================================================================

class TransformerAttentionLayer:
    """
    Complete attention layer used in Transformers (GPT, BERT, etc.)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)

        # FFN (Feed-Forward Network)
        if d_ff is None:
            d_ff = d_model * 4

        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

        # Layer normalization parameters (would be learned in practice)
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def layer_norm(self, x, gamma, beta, eps=1e-6):
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def forward(self, x, mask=None):
        """
        Complete forward pass with:
        1. Multi-head self-attention
        2. Residual connection + Layer norm
        3. Feed-forward network
        4. Residual connection + Layer norm
        """

        # Self-attention with residual
        attn_out, _ = self.attention.forward(x, x, x, mask)
        x = x + attn_out  # Residual connection
        x = self.layer_norm(x, self.gamma1, self.beta1)  # Post-attention norm

        # Feed-forward network
        ffn_out = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU activation
        ffn_out = np.dot(ffn_out, self.W2) + self.b2

        # Residual + norm
        x = x + ffn_out
        x = self.layer_norm(x, self.gamma2, self.beta2)

        return x

def transformer_attention_demo():
    """Full transformer attention layer demonstration."""

    print("\n" + "-" * 50)
    print("TRANSFORMER ATTENTION LAYER")
    print("-" * 50)

    seq_len, d_model, num_heads = 6, 8, 2

    # Random input
    x = np.random.randn(1, seq_len, d_model)

    print(f"Input shape: {x.shape}")
    print(f"Model dimension: {d_model}")
    print(f"Attention heads: {num_heads}")
    print(f"FFN hidden dimension: {d_model * 4}")

    # Create and run layer
    layer = TransformerAttentionLayer(d_model, num_heads)
    output = layer.forward(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"(Same as input - transformer layers preserve dimensions)")

    print("\n" + "-" * 50)
    print("TRANSFORMER LAYER FLOW")
    print("-" * 50)
    print("""
    Input x (seq_len, d_model)
           │
           ▼
    ┌──────┴──────┐
    │   Multi-Head  │
    │  Self-Attention│
    └──────┬──────┘
           │
           ▼
    x + Attention ──► Residual Connection
           │
           ▼
    LayerNorm ────────► Post-Attention Norm
           │
           ▼
    ┌──────┴──────┐
    │  Feed-Forward │
    │   (ReLU→Linear) │
    └──────┬──────┘
           │
           ▼
    x + FFN ──► Residual Connection
           │
           ▼
    LayerNorm ────────► Post-FFN Norm
           │
           ▼
    Output (same dim as input)
    """)

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  PART 1: ATTENTION INTUITION")
    print("=" * 70)
    attention_intuition()

    print("\n" + "=" * 70)
    print("  PART 2: SCALED DOT-PRODUCT ATTENTION")
    print("=" * 70)
    attn_weights = scaled_dot_product_attention()
    visualize_attention()

    print("\n" + "=" * 70)
    print("  PART 3: MULTI-HEAD ATTENTION")
    print("=" * 70)
    multi_head_demo()

    print("\n" + "=" * 70)
    print("  PART 4: CAUSAL MASKING")
    print("=" * 70)
    causal_masking()

    print("\n" + "=" * 70)
    print("  PART 5: COMPLETE LAYER")
    print("=" * 70)
    transformer_attention_demo()

    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. ATTENTION allows tokens to look at all other tokens
       - Q, K, V projections enable learned "what to attend to"
       - Scaled dot-product is computationally efficient

    2. MULTI-HEAD ATTENTION runs multiple attention in parallel
       - Each head learns different relationship types
       - Concatenated output captures diverse information

    3. CAUSAL MASKING prevents seeing future tokens
       - Essential for autoregressive models (GPT)
       - Encoder doesn't need this (sees full sequence)

    4. RESIDUAL + LAYER NORM ensure stable training
       - Skip connections help gradient flow
       - Layer norm stabilizes activations

    NEXT: Build a complete Transformer from scratch
    """)

    print("\n" + "=" * 70)
    print("  EXERCISES")
    print("=" * 70)
    print("""
    1. Implement scaled dot-product attention with batching

    2. Visualize attention patterns for a real sentence

    3. Implement relative position embeddings

    4. Build flash attention (IO-aware attention)

    5. Compare attention patterns across layers
    """)
