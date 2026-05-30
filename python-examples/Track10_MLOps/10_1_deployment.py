"""
Track 10.1: LLM Deployment with vLLM
=====================================
Production inference with vLLM, quantization, and optimization.

Author: AI Engineering Masterclass
"""

# ==============================================================================
# PART 1: VLLM BASICS
# ==============================================================================

def vllm_server():
    """Using vLLM for high-performance inference."""
    try:
        from vllm import LLM, SamplingParams

        # Initialize model
        llm = LLM(model="meta-llama/Llama-3-8B-Instruct")

        # Configure sampling
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256
        )

        # Generate
        outputs = llm.generate(["Hello, how are you?", "What is AI?"], sampling_params)

        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")
            print()

    except ImportError:
        print("Install: pip install vllm")

def vllm_batch_inference():
    """Batch inference for efficiency."""
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
        sampling = SamplingParams(temperature=0.7, max_tokens=128)

        # Process multiple prompts at once
        prompts = [
            "Explain quantum computing",
            "What is machine learning?",
            "Write a Python function",
            "Summarize this: AI is transforming industries",
            "Write a haiku about technology"
        ]

        outputs = llm.generate(prompts, sampling)

        for i, output in enumerate(outputs):
            print(f"[{i+1}] {prompts[i][:30]}... → {output.outputs[0].text[:50]}...")

    except ImportError:
        print("vLLM not installed")

# ==============================================================================
# PART 2: QUANTIZATION
# ==============================================================================

def quantization_comparison():
    """Compare quantization levels and their impact."""

    configs = {
        "FP16": {"precision": "16-bit float", "size_gb": 16, "quality": 1.0},
        "INT8": {"precision": "8-bit int", "size_gb": 8, "quality": 0.95},
        "INT4": {"precision": "4-bit int", "size_gb": 4, "quality": 0.85},
        "GPTQ": {"precision": "4-bit GPTQ", "size_gb": 4.5, "quality": 0.90},
        "AWQ": {"precision": "4-bit AWQ", "size_gb": 4.2, "quality": 0.92},
        "GGUF": {"precision": "4-bit GGUF", "size_gb": 4.0, "quality": 0.88}
    }

    print("Quantization Comparison (for 7B model):")
    print("-" * 50)
    for name, info in configs.items():
        print(f"{name:8s} | {info['size_gb']} GB | Quality: {info['quality']:.0%}")

    return configs

# ==============================================================================
# PART 3: OLLAMA LOCAL INFERENCE
# ==============================================================================

def ollama_local():
    """Using Ollama for local inference."""
    try:
        import ollama

        # List available models
        models = ollama.list()
        print(f"Available models: {[m['name'] for m in models['models']]}")

        # Generate response
        response = ollama.chat(model='llama3', messages=[
            {"role": "user", "content": "What is RAG?"}
        ])
        print(f"Response: {response['message']['content']}")

        # Stream response
        stream = ollama.chat(model='llama3', messages=[
            {"role": "user", "content": "Write a short story"}
        ], stream=True)

        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)

    except ImportError:
        print("Install: pip install ollama")

# ==============================================================================
# PART 4: LANGCHAIN INTEGRATION
# ==============================================================================

def langchain_llm():
    """Using LLMs with LangChain."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage

        # OpenAI
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        response = llm([HumanMessage(content="Hello!")])
        print(f"ChatGPT: {response.content}")

        # With streaming
        for chunk in llm.stream("Count to 5"):
            print(chunk.content, end='', flush=True)

    except ImportError:
        print("Install: pip install langchain-openai")

# ==============================================================================
# PART 5: MONITORING & OBSERVABILITY
# ==============================================================================

def langsmith_example():
    """Setting up LangSmith for observability."""
    import os

    # Configure LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
    os.environ["LANGCHAIN_PROJECT"] = "ai-engineering-masterclass"

    try:
        from langchain_openai import ChatOpenAI
        from langchain.callbacks.tracing import LangChainTracer

        llm = ChatOpenAI(model="gpt-4")
        tracer = LangChainTracer()

        # All calls will be traced!
        response = llm.invoke("Hello", config={"callbacks": [tracer]})
        print(f"Traced response: {response}")

    except Exception as e:
        print(f"LangSmith setup: {e}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TRACK 10.1: LLM DEPLOYMENT")
    print("=" * 70)

    print("\n1. Quantization Comparison:")
    quantization_comparison()

    print("\n" + "=" * 70)
    print("  KEY DEPLOYMENT OPTIONS")
    print("=" * 70)
    print("""
    1. vLLM - High throughput, continuous batching
       pip install vllm

    2. Ollama - Easy local deployment
       ollama run llama3

    3. HuggingFace TGI - Production inference server
       docker run -p 8080:80 ghcr.io/huggingface/text-generation-inference

    4. Cloud APIs - OpenAI, Anthropic, Google
       - Easiest, highest cost
       - No infrastructure management

    5. Quantization options:
       - GGUF (llama.cpp) - Best for local
       - GPTQ/AWQ - Best for GPU servers
       - INT8/INT4 - Balance of speed and quality
    """)

    print("\n" + "=" * 70)
    print("  NEXT: Agentic AI - LangGraph & CrewAI")
    print("=" * 70)