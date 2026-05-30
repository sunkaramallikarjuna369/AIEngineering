"""
Track 5.1: Frontier LLM APIs
=============================
Using OpenAI GPT-4, Anthropic Claude, and Google Gemini APIs.
Streaming responses, async calls, error handling.

Author: AI Engineering Masterclass
"""

import os
import time
from typing import List, Dict, Optional

# ==============================================================================
# PART 1: OPENAI API
# ==============================================================================

def openai_basic():
    """Basic OpenAI API usage."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Simple completion
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is RAG in 2 sentences?"}
        ],
        temperature=0.7,
        max_tokens=150
    )

    print("OpenAI Response:")
    print(response.choices[0].message.content)
    print(f"Tokens used: {response.usage.total_tokens}")

def openai_streaming():
    """Streaming responses for better UX."""
    from openai import OpenAI

    client = OpenAI()

    print("Streaming response:")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

# ==============================================================================
# PART 2: ANTHROPIC API
# ==============================================================================

def anthropic_basic():
    """Basic Anthropic Claude API usage."""
    from anthropic import Anthropic

    client = Anthropic()

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Explain transformer attention in one paragraph."}
        ]
    )

    print("Claude Response:")
    print(message.content[0].text)
    print(f"Tokens: input={message.usage.input_tokens}, output={message.usage.output_tokens}")

def anthropic_streaming():
    """Anthropic streaming with thinking."""
    from anthropic import Anthropic

    client = Anthropic()

    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write a short poem about AI"}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()

# ==============================================================================
# PART 3: GOOGLE GEMINI API
# ==============================================================================

def gemini_basic():
    """Google Gemini API usage."""
    from google.generativeai import GenerativeModel

    model = GenerativeModel('gemini-2.0-flash')

    response = model.generate_content("What is a neural network?")

    print("Gemini Response:")
    print(response.text)

# ==============================================================================
# PART 4: UNIFIED API WITH LITELLM
# ==============================================================================

def litellm_unified():
    """Using LiteLLM for unified API access to multiple providers."""
    try:
        from litellm import completion

        # OpenAI
        response = completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"OpenAI: {response['choices'][0]['message']['content']}")

        # Anthropic (same interface!)
        response = completion(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"Anthropic: {response['choices'][0]['message']['content']}")

    except ImportError:
        print("Install litellm: pip install litellm")

# ==============================================================================
# PART 5: MODEL COMPARISON
# ==============================================================================

def compare_models():
    """Compare responses from different models."""
    from openai import OpenAI
    from anthropic import Anthropic

    openai_client = OpenAI()
    anthropic_client = Anthropic()

    prompt = "Explain what a vector database is to a 5 year old."

    # GPT-4o
    gpt_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("GPT-4o:", gpt_response.choices[0].message.content[:100], "...")

    # Claude
    claude_response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    print("Claude:", claude_response.content[0].text[:100], "...")

# ==============================================================================
# PART 6: FUNCTION CALLING
# ==============================================================================

def function_calling():
    """Using function calling with OpenAI."""
    from openai import OpenAI

    client = OpenAI()

    # Define functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=tools
    )

    # Check if model wants to call a function
    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"Function to call: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TRACK 5.1: FRONTIER LLM APIs")
    print("=" * 70)

    # Note: Requires API keys in environment variables
    # export OPENAI_API_KEY=sk-...
    # export ANTHROPIC_API_KEY=sk-ant-...

    print("\nSet your API keys as environment variables:")
    print("  export OPENAI_API_KEY=sk-...")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")

    print("\nThen uncomment the functions above to run them.")