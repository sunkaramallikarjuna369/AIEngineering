"""
Track 4.1: Attention Mechanism - Super Simple Edition
=====================================================
Understanding how attention works in transformers.

Author: AI Engineering Masterclass
"""

print("=" * 60)
print("  WHAT IS ATTENTION?")
print("=" * 60)

print("""
[LOOK] ATTENTION = What to focus on

Imagine reading: "The cat sat on the mat"

When processing "sat":
- You focus on "cat" (the subject)
- You also notice "mat" (the object)
- You ignore "The" (less important)

[BOT] ATTENTION MECHANISM does the SAME thing!

   For each word, the model decides:
   - Which other words are important?
   - How much to "pay attention" to each?

--------------------------------------------------------------------------------

   THE MATH (simplified):

   1. Compare query with all keys
   2. Get scores (similarity)
   3. Softmax to get weights (0-1, sum to 1)
   4. Weighted sum of values

--------------------------------------------------------------------------------
""")

# ==============================================================================
# PART 1: ATTENTION INTUITION
# ==============================================================================

print("\n" + "=" * 60)
print("  ATTENTION ANALOGY: LIBRARY SEARCH")
print("=" * 60)

print("""
   [BOOK] Think of attention like searching a library:

   [KEY] KEYS = What's in each book (book titles, topics)
   [?] QUERY = What you're looking for
   [PAGE] VALUES = The actual book content

   When you search:
   1. Your query matches against book titles
   2. Best matches get highest scores
   3. You retrieve the relevant book content

   ATTENTION = Same thing, but for words!
""")

# ==============================================================================
# PART 2: SIMPLE ATTENTION EXAMPLE
# ==============================================================================

import numpy as np

def simple_attention():
    """Calculate attention between words."""
    print("\n[NUM] SIMPLE ATTENTION CALCULATION")
    print("-" * 40)

    # Sentence: "The cat sat on the mat"
    words = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(words)

    # Simple similarity scores (who relates to whom)
    # Row = query word, Column = key word, Value = how related
    # Higher = more attention
    similarity_scores = np.array([
        [0.3, 0.4, 0.2, 0.2, 0.3, 0.1],  # "The"
        [0.4, 0.8, 0.5, 0.1, 0.4, 0.3],  # "cat"
        [0.2, 0.5, 0.7, 0.4, 0.3, 0.6],  # "sat"
        [0.2, 0.1, 0.4, 0.6, 0.5, 0.3],  # "on"
        [0.3, 0.4, 0.2, 0.5, 0.3, 0.2],  # "the"
        [0.1, 0.3, 0.6, 0.3, 0.2, 0.8],  # "mat"
    ])

    print("Sentence: 'The cat sat on the mat'\n")
    print("Who pays attention to whom? (higher = more attention)\n")

    for i, word in enumerate(words):
        print(f"'{word}' attends most to:", end=" ")
        top_idx = np.argmax(similarity_scores[i])
        print(f"'{words[top_idx]}' (score: {similarity_scores[i][top_idx]:.1f})")

    return similarity_scores

scores = simple_attention()

# ==============================================================================
# PART 3: ATTENTION FORMULA
# ==============================================================================

print("\n" + "=" * 60)
print("  THE ATTENTION FORMULA")
print("=" * 60)

print("""
   [CHART] THE FORMULA:

   Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V

   Where:
   - Q = Query (what I'm looking for)
   - K = Key (what I contain)
   - V = Value (actual information)
   - d = dimension (scaling factor)
   - softmax = turns scores to probabilities
""")

def attention_formula():
    """Demonstrate the attention formula."""
    print("\n[CALC] ATTENTION FORMULA STEP-BY-STEP")

    # Step 1: Q, K, V matrices (simplified)
    # For 3 words, each with 4-dimensional embedding
    Q = np.array([[0.1, 0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7, 0.8],
                  [0.2, 0.1, 0.9, 0.3]])

    K = np.array([[0.1, 0.5, 0.2, 0.3],
                  [0.4, 0.2, 0.6, 0.7],
                  [0.3, 0.1, 0.4, 0.5]])

    V = np.array([[0.5, 0.3, 0.8, 0.2],
                  [0.1, 0.7, 0.4, 0.6],
                  [0.6, 0.2, 0.5, 0.9]])

    print("Step 1: QK^T (similarity between queries and keys)")
    QK = np.dot(Q, K.T)
    print(f"   Shape: {QK.shape}")
    print(f"   Values:\n{QK.round(3)}")

    print("\nStep 2: Scale by √d (prevents large values)")
    d = Q.shape[1]
    scaled = QK / np.sqrt(d)
    print(f"   √d = √{d} = {np.sqrt(d):.2f}")
    print(f"   Scaled scores:\n{scaled.round(3)}")

    print("\nStep 3: Softmax (turn to probabilities)")
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_weights = softmax(scaled)
    print(f"   Weights (each row sums to 1):\n{attention_weights.round(3)}")

    print("\nStep 4: Softmax × V (weighted sum)")
    output = np.dot(attention_weights, V)
    print(f"   Final output:\n{output.round(3)}")
    print(f"   Shape: {output.shape}")

attention_formula()

# ==============================================================================
# PART 4: WHY MULTI-HEAD?
# ==============================================================================

print("\n" + "=" * 60)
print("  MULTI-HEAD ATTENTION")
print("=" * 60)

print("""
   [BRAIN] MULTIPLE HEADS = Multiple perspectives

   Instead of one attention, use several in parallel!
   Each head learns different relationships:

   Head 1: Subject-verb relationships
   "cat" -> "sat" (cat performs the action)

   Head 2: Proximity (words near each other)
   "the" -> "mat" (articles are close to nouns)

   Head 3: Semantic similarity
   "sat" -> "mat" (both are related to sitting)

   Then combine all heads together!
""")

def multi_head_demo():
    """Simple multi-head attention visualization."""
    print("\n[TARGET] MULTI-HEAD ATTENTION EXAMPLE")
    print("-" * 40)

    print("""
   Sentence: "The cat sat on the mat"

   Head 1 (Subject-Verb):
   cat → sat (cat is doing the sitting)

   Head 2 (Determiner-Noun):
   the → cat (the relates to cat)
   the → mat (the relates to mat)

   Head 3 (Proximity):
   sat → mat (sat is near mat)
   sat → on (on is the preposition)

   Combined = Rich understanding!
   """)

multi_head_demo()

# ==============================================================================
# PART 5: TRANSFORMER ATTENTION
# ==============================================================================

print("\n" + "=" * 60)
print("  TRANSFORMER ATTENTION")
print("=" * 60)

print("""
   [FLASH] TRANSFORMER = Stacked Attention Layers

   +----------------------------------+
   |  Input: "The cat sat on the mat" |
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |  Self-Attention Layer 1          |
   |  (each word attends to all)      |
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |  Self-Attention Layer 2          |
   |  (higher-level patterns)         |
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |  Feed-Forward Network            |
   |  (more processing)               |
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |  Output: Rich representations    |
   +----------------------------------+

   Key Innovation: All words can "talk" to all other words!
   (No sequential processing like RNNs)
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
[OK] Attention = What to focus on

[OK] Query, Key, Value = Search system for words

[OK] Attention(Q,K,V) = How much to attend * what to retrieve

[OK] Softmax = Turn scores to probabilities (0-1)

[OK] Multi-Head = Multiple perspectives on relationships

[OK] Transformer = Stack many attention layers

[OK] Why important: Foundation of GPT, BERT, Claude, etc.

NEXT: Build a transformer from scratch!
""")