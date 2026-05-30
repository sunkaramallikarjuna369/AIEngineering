"""
Track 6: Prompt Engineering
===========================
Mastering prompt techniques: zero-shot, few-shot, chain-of-thought, etc.

Author: AI Engineering Masterclass
"""

from typing import List, Dict

# ==============================================================================
# PART 1: BASIC PROMPTING
# ==============================================================================

def zero_shot_prompting():
    """
    Zero-shot: No examples, just direct instruction.
    Works well for simple, clear tasks.
    """
    prompt = """Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL:

Review: "This product exceeded my expectations. Highly recommended!"
Sentiment:"""

    return prompt

def one_shot_prompting():
    """
    One-shot: One example to guide the model.
    Better for tasks where format matters.
    """
    prompt = """Classify the sentiment of product reviews.

Example:
Review: "Terrible quality, broke after one day."
Sentiment: NEGATIVE

Now classify:
Review: "Amazing value for the price!"
Sentiment:"""

    return prompt

def few_shot_prompting():
    """
    Few-shot: Multiple examples for better understanding.
    Great for complex patterns or domain-specific responses.
    """
    prompt = """Translate English to French:

English: "Hello, how are you?"
French: "Bonjour, comment allez-vous?"

English: "Where is the nearest airport?"
French: "Où est l'aéroport le plus proche?"

English: "I would like to book a room"
French:"""

    return prompt

# ==============================================================================
# PART 2: CHAIN-OF-THOUGHT
# ==============================================================================

def chain_of_thought():
    """
    Chain-of-Thought: Ask model to show reasoning steps.
    Dramatically improves complex reasoning tasks.
    """
    prompt = """Solve this problem step by step:

Problem: If a store has 45 apples and sells 23 in the morning and 12 in the afternoon, how many apples remain?

Let me work through this step by step:
1. Starting apples: 45
2. Sold in morning: 23
3. Remaining after morning: 45 - 23 = 22
4. Sold in afternoon: 12
5. Remaining after afternoon: 22 - 12 = 10

Answer: 10 apples remain.

---

Now solve this:
Problem: A train travels 120 miles in 2 hours, then stops for 30 minutes, then travels 80 miles in 1.5 hours. What was the average speed?

Let me work through this step by step:"""

    return prompt

def self_consistency():
    """
    Self-consistency: Generate multiple solutions, take majority vote.
    Reduces errors in complex reasoning.
    """
    prompt_template = """Solve this problem and show your work:

Problem: {problem}

Generate 3 different solutions and check if they agree.
If they disagree, explain why and determine the correct answer.
"""

    return prompt_template

# ==============================================================================
# PART 3: STRUCTURED OUTPUT
# ==============================================================================

def json_mode_prompt():
    """Get structured JSON output."""
    prompt = """Extract information about the person from the text below.

Text: "John Smith is a software engineer at Google with 5 years of experience. He graduated from MIT in 2019 with a degree in Computer Science. He lives in San Francisco."

Return the information as JSON with these fields:
- name
- job_title
- company
- years_experience
- university
- graduation_year
- city

Format your response as valid JSON only, no other text."""

    return prompt

def xml_format_prompt():
    """Use XML tags for structured output."""
    prompt = """List the top 3 benefits of exercise.

Format your response using XML tags:
<benefits>
  <benefit>
    <name>Benefit name here</name>
    <description>Description here</description>
  </benefit>
</benefits>"""

    return prompt

# ==============================================================================
# PART 4: SYSTEM PROMPTS
# ==============================================================================

def system_prompt_examples():
    """Different system prompt styles."""

    # Style 1: Direct role
    role_prompt = """System: You are an experienced Python programmer with 10 years of experience.
User: Write a function to find prime numbers."""

    # Style 2: Constrained role
    constrained_prompt = """System: You are a safety-focused AI assistant. Before providing any code, you must:
1. Check for potential security vulnerabilities
2. Consider edge cases
3. Add appropriate error handling
4. Include documentation

User: Write a function to connect to a database."""

    # Style 3: Persona
    persona_prompt = """System: You are a friendly and patient tutor who explains concepts using simple analogies.
You never use jargon without explaining it first.
When the user seems confused, you provide additional examples.

User: What is recursion?"""

    # Style 4: Constitutional (Anthropic style)
    constitutional_prompt = """System: You are a helpful AI assistant.
Your goals:
- Be helpful but harmless
- Be honest about limitations
- Respect privacy and confidentiality

User: How do I hack into my neighbor's WiFi?"""

    return {
        "role": role_prompt,
        "constrained": constrained_prompt,
        "persona": persona_prompt,
        "constitutional": constitutional_prompt
    }

# ==============================================================================
# PART 5: PROMPT PATTERNS
# ==============================================================================

def prompt_patterns():
    """Common reusable prompt patterns."""

    patterns = {
        "persona": """Act as a [ROLE].
[CONTEXT/QUESTION]

Take deep breath and answer carefully.""",

        "template": """Given the following [INPUT_TYPE]:
[INPUT]

[REQUIREMENTS]:
- [Requirement 1]
- [Requirement 2]

Provide a [OUTPUT_TYPE] that meets these requirements.""",

        "chain": """Task: [MAIN_TASK]

Sub-tasks:
1. [Sub-task 1]
2. [Sub-task 2]
3. [Sub-task 3]

Complete each sub-task, then synthesize into a final answer.""",

        "verification": """Statement: [STATEMENT]
Question: Is this statement true?

Verify by:
1. Checking sources
2. Identifying assumptions
3. Evaluating evidence
4. Drawing conclusion""",

        "refusal_override": """Base response pattern for harmful requests:
"I'm not able to help with that because [REASON]. However, I'd be happy to help with [RELATED_HELPFUL_TASK] instead."

Harmful request: [USER_INPUT]

Provide refusal with helpful redirect."""
    }

    return patterns

# ==============================================================================
# PART 6: TEMPERATURE & SAMPLING
# ==============================================================================

def sampling_params():
    """Understanding temperature, top_p, top_k."""

    configs = {
        # Deterministic - best for factual Q&A
        "factual": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1
        },

        # Balanced - good general purpose
        "balanced": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        },

        # Creative - for brainstorming
        "creative": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 100
        },

        # Code - precise but not rigid
        "code": {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50
        }
    }

    return configs

# ==============================================================================
# PRACTICAL EXAMPLES
# ==============================================================================

def practical_examples():
    """Real-world prompt engineering examples."""

    examples = {
        "email_summarization": """Summarize this email in exactly 3 bullet points:
- First bullet: main topic/request
- Second bullet: key details or deadlines
- Third bullet: required action (if any)

Email:
{email_content}""",

        "code_review": """Review the following code for:
1. Security vulnerabilities
2. Performance issues
3. Code quality concerns
4. Potential bugs

Format findings as:
### Issue [N]: [Title]
**Severity:** [Critical/High/Medium/Low]
**Location:** [File:Line]
**Description:** [Explanation]
**Recommendation:** [Fix suggestion]

Code:
{code}""",

        "data_extraction": """Extract structured data from the following text.

For each entity found, extract:
- entity_type (PERSON, ORGANIZATION, LOCATION, DATE, MONEY)
- value
- context

Text: {text}

Output as JSON array:
{entities: [{type, value, context}]}""",

        "classification_with_confidence": """Classify this review and rate your confidence:

Review: "{review}"

Classes: POSITIVE, NEGATIVE, NEUTRAL, MIXED

Respond in format:
Classification: [CLASS]
Confidence: [0-100]%
Reasoning: [2 sentence explanation]"""
    }

    return examples

# ==============================================================================
# MAIN / DEMO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TRACK 6: PROMPT ENGINEERING")
    print("=" * 70)

    print("\n1. Zero-Shot Prompting:")
    print(zero_shot_prompting())

    print("\n" + "-" * 50)
    print("\n2. Chain-of-Thought:")
    print(chain_of_thought())

    print("\n" + "-" * 50)
    print("\n3. JSON Mode Prompt:")
    print(json_mode_prompt())

    print("\n" + "-" * 50)
    print("\n4. Prompt Patterns:")
    patterns = prompt_patterns()
    for name, pattern in patterns.items():
        print(f"\n{name.upper()}:")
        print(pattern[:100] + "...")

    print("\n" + "=" * 70)
    print("  NEXT: Try these prompts with GPT-4 or Claude!")
    print("=" * 70)