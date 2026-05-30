"""
Track 5.1: LLM Basics - Super Simple Edition
=============================================
What are LLMs? How do they work? Simple explanations.

Author: AI Engineering Masterclass
"""

print("=" * 60)
print("  WHAT ARE LARGE LANGUAGE MODELS?")
print("=" * 60)

print("""
🤖 LLM = Large Language Model

   Think of it like this:

   📚 You've read MILLIONS of books, articles, websites
   📝 You've learned patterns: grammar, facts, reasoning
   💬 I can now CHAT with you using what I learned

   That's what an LLM does!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   EXAMPLES OF LLMs:
   - GPT-4, GPT-4o (OpenAI)
   - Claude 3.5 (Anthropic)
   - Gemini (Google)
   - Llama (Meta)
   - Mistral (Mistral AI)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   WHAT CAN THEY DO?
   ✓ Write essays, emails, code
   ✓ Answer questions
   ✓ Translate languages
   ✓ Summarize documents
   ✓ Reason through problems
""")

# ==============================================================================
# PART 1: HOW LLMs LEARN - PRETRAINING
# ==============================================================================

print("\n" + "=" * 60)
print("  HOW LLMs LEARN (Pre-training)")
print("=" * 60)

print("""
   📚 STEP 1: READ EVERYTHING ON THE INTERNET

   The model reads:
   - Billions of web pages
   - Books and articles
   - Code from GitHub
   - Wikipedia
   - Reddit, Twitter, etc.

   That's like reading 100,000 lifetimes of reading!

   🧠 STEP 2: LEARN PATTERNS

   After reading so much, the model learns:
   - Grammar rules (how to form sentences)
   - Facts (Paris is the capital of France)
   - Reasoning (if A then B)
   - Common sense (fire is hot)
   - Coding patterns (how to write Python)

   This is called UNSUPERVISED LEARNING
   Just predict the next word in a sentence!
""")

def word_prediction_game():
    """Demonstrate next-word prediction."""
    print("\n🎯 THE WORD PREDICTION GAME")
    print("-" * 40)

    sentences = [
        ("The sky is", "blue"),
        ("Python is a", "programming"),
        ("To make coffee, you need to", "brew"),
        ("After raining, you see a", "rainbow"),
    ]

    print("Can you predict the next word?\n")
    for start, answer in sentences:
        print(f"   '{start} ___'")
        print(f"   → '{answer}' ✓\n")

word_prediction_game()

# ==============================================================================
# PART 2: TOKENIZATION
# ==============================================================================

print("\n" + "=" * 60)
print("  TOKENIZATION (How LLMs Read)")
print("=" * 60)

print("""
   🔤 LLMs don't read letters or words
   They read TOKENS

   Token = A piece of a word (or whole word)

   Example:
   "Hello, world!" might become:

   "Hello" → token 1234
   ","    → token 56
   "world" → token 7890
   "!"    → token 12

   Why? Computer memory! Tokens are numbers.
   1 token ≈ 4 characters of English text
   ~750 words ≈ 1000 tokens
""")

def tokenization_demo():
    """Show how text becomes tokens."""
    print("\n🔢 TOKENIZATION EXAMPLE")

    # Simplified tokenization
    text = "Hello, how are you?"

    # Rough token estimation
    # Each word ~1 token, punctuation ~0.2 tokens
    words = text.split()
    tokens_approx = len(words) + int(len(text) * 0.1)

    print(f"   Text: '{text}'")
    print(f"   Words: {words}")
    print(f"   Estimated tokens: ~{tokens_approx}")

    print("\n   More examples:")
    examples = [
        "AI is amazing",
        "The quick brown fox jumps",
        "I love learning about machine learning"
    ]
    for text in examples:
        word_count = len(text.split())
        token_est = word_count + int(len(text) * 0.1)
        print(f"   '{text}' → ~{token_est} tokens")

tokenization_demo()

# ==============================================================================
# PART 3: CONTEXT WINDOW
# ==============================================================================

print("\n" + "=" * 60)
print("  CONTEXT WINDOW")
print("=" * 60)

print("""
   📏 CONTEXT WINDOW = How much text the model can see at once

   Think of it like a "working memory"

   Example:
   - GPT-3: 2,000 tokens (short essay)
   - GPT-4: 128,000 tokens (novel)
   - Claude 3.5: 200,000 tokens (long book)
   - Gemini 1.5: 1,000,000 tokens (entire codebase!)

   ⚠️ LIMITATION: Old text gets "forgotten"
   If window is 4,000 tokens and you chat for hours,
   early messages might be forgotten
""")

# ==============================================================================
# PART 4: TEMPERATURE & SAMPLING
# ==============================================================================

print("\n" + "=" * 60)
print("  TEMPERATURE (How Creative?)")
print("=" * 60)

print("""
   🌡️ TEMPERATURE = How random/creative the output is

   Low temperature (0.1-0.3):
   - Same input → almost same output
   - Good for: facts, math, precise answers
   - Like a textbook

   Medium temperature (0.5-0.7):
   - Some variety in responses
   - Good for: general conversation
   - Like a friendly chat

   High temperature (0.9-1.0):
   - Very creative, surprising outputs
   - Good for: brainstorming, stories
   - Like an imaginative writer
""")

def temperature_demo():
    """Visualize temperature effect."""
    print("\n🌡️ TEMPERATURE EFFECT")

    print("""
   Question: "What is the color of the sky?"

   Temperature 0.1 (Deterministic):
   → "The sky is blue."
   (Always the same factual answer)

   Temperature 0.5 (Balanced):
   → "The sky appears blue during clear days."
   (Factual with slight variation)

   Temperature 1.0 (Creative):
   → "The sky is a vast blue canvas painted by nature,
      sometimes streaked with clouds like brushstrokes."
   (More poetic and varied)
    """)

temperature_demo()

# ==============================================================================
# PART 5: FINE-TUNING vs RAG vs PROMPTING
# ==============================================================================

print("\n" + "=" * 60)
print("  3 WAYS TO MAKE LLMs MORE USEFUL")
print("=" * 60)

print("""
   1️⃣ PROMPTING (Cheapest, Fastest)
   ────────────────────────────────
   Write better instructions in the prompt

   Example:
   "You are a helpful coding assistant..."

   Best for: Simple customizations

   ─────────────────────────────────────────

   2️⃣ RAG - Retrieval Augmented Generation
   ──────────────────────────────────────────
   Give the model documents to use as reference

   Example:
   "Based on this document, answer: [your question]"
   + [attach company policy document]

   Best for: Adding specific knowledge (like your docs)

   ─────────────────────────────────────────

   3️⃣ FINE-TUNING (Most Expensive, Permanent)
   ───────────────────────────────────────────
   Train the model on custom data

   Example:
   Fine-tune on 10,000 customer service chats
   → Model learns your brand voice permanently

   Best for: Changing how the model speaks/responds
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
✅ LLM = Trained on massive text data

✅ Learns by predicting next word (self-supervised)

✅ Tokens = Pieces of text (model reads in numbers)

✅ Context window = How much it can "see" at once

✅ Temperature = How creative vs precise

✅ 3 ways to customize:
   - Prompting (instructions)
   - RAG (documents)
   - Fine-tuning (training)

NEXT: Learn how to call LLM APIs in code!
""")