"""
Track 9.1: AI Agents Basics - Super Simple Edition
==================================================
What are AI Agents? How do they use tools and take actions?

Author: AI Engineering Masterclass
"""

print("=" * 60)
print("  WHAT ARE AI AGENTS?")
print("=" * 60)

print("""
🤖 AI AGENT = LLM + Memory + Tools + Actions

   Think of it like a helpful robot assistant:

   🤖 Robot Brain (LLM)
   └─ Makes decisions, understands requests

   📚 Memory
   └─ Remembers past interactions

   🔧 Tools (can use)
   └─ Search the web, run code, read files

   🎯 Actions
   └─ Actually does things (books flight, sends email)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   AGENT vs SIMPLE CHATBOT:

   🤖 Simple Chatbot:
   "What's the weather?"
   → "It's sunny today."

   🤖 Agent:
   "Book me a flight to NYC next Friday"
   → Searches flights → Compares prices → Books ticket
""")

# ==============================================================================
# PART 1: THE AGENT LOOP
# ==============================================================================

print("\n" + "=" * 60)
print("  THE AGENT LOOP")
print("=" * 60)

print("""
   🔄 REACT LOOP (Reason + Act + Observe):

   ┌─────────────────────────────────────────┐
   │  1. OBSERVE                             │
   │     What does the user want?            │
   └──────────────────┬──────────────────────┘
                      ↓
   ┌─────────────────────────────────────────┐
   │  2. REASON                              │
   │     Think: What should I do first?      │
   └──────────────────┬──────────────────────┘
                      ↓
   ┌─────────────────────────────────────────┐
   │  3. ACT                                 │
   │     Use a tool or give response         │
   └──────────────────┬──────────────────────┘
                      ↓
   ┌─────────────────────────────────────────┐
   │  4. OBSERVE RESULT                      │
   │     What did the tool return?           │
   └──────────────────┬──────────────────────┘
                      ↓
                    REPEAT until done!
""")

def agent_loop_demo():
    """Demonstrate the agent reasoning loop."""
    print("\n🔄 AGENT LOOP EXAMPLE")
    print("-" * 40)

    task = "What is the weather in Tokyo and should I bring an umbrella?"

    print(f"Task: '{task}'\n")

    steps = [
        ("OBSERVE", "User wants to know Tokyo weather + packing advice"),
        ("REASON", "I need to search for Tokyo weather first"),
        ("ACT", "Calling weather tool..."),
        ("OBSERVE", "Result: 22°C, rain expected, 80% chance"),
        ("REASON", "It's rainy! User should bring an umbrella"),
        ("ACT", "Providing answer with weather + recommendation"),
        ("DONE", "Final response delivered"),
    ]

    for i, (step, thought) in enumerate(steps, 1):
        emoji = {"OBSERVE": "👁️", "REASON": "🤔", "ACT": "🔧", "DONE": "✅"}.get(step, "➡️")
        print(f"   Step {i}: {emoji} {step}")
        print(f"      {thought}\n")

agent_loop_demo()

# ==============================================================================
# PART 2: TOOLS
# ==============================================================================

print("\n" + "=" * 60)
print("  TOOLS (What Agents Can Use)")
print("=" * 60)

print("""
   🔧 TOOLS extend what agents can do:

   🌐 SEARCH
   │ Browse the web, find information
   │ Example: "Search for iPhone 15 prices"
   │
   💻 CODE
   │ Run Python, execute calculations
   │ Example: "Calculate compound interest"
   │
   📁 FILES
   │ Read/write documents, PDFs
   │ Example: "Summarize this PDF"
   │
   📧 EMAIL
   │ Send messages, check inbox
   │ Example: "Email the report to John"
   │
   📊 APIs
   │ Call external services
   │ Example: "Check my bank balance"
   │
   🔍 SEARCH (Vector)
   │ Query document database
   │ Example: "Find our return policy"
""")

# ==============================================================================
# PART 3: BUILDING A SIMPLE AGENT
# ==============================================================================

print("\n" + "=" * 60)
print("  BUILDING A SIMPLE AGENT")
print("=" * 60)

def simple_agent():
    """Simple agent with tools."""
    print("\n🚀 SIMPLE AGENT CODE")
    print("-" * 40)

    # Define tools
    tools = {
        "calculator": lambda expr: eval(expr),
        "search": lambda query: f"Found: {query} - Top result from Wikipedia",
        "weather": lambda city: f"22°C, partly cloudy in {city}",
    }

    # Agent's knowledge (what it can do)
    tool_descriptions = {
        "calculator": "Use for math: calculator(expression)",
        "search": "Use to find info: search(query)",
        "weather": "Use for weather: weather(city)",
    }

    def agent(query, max_steps=5):
        print(f"\n📝 Task: {query}\n")

        # Simple rule-based decision
        if "calculate" in query.lower() or "+" in query or "-" in query:
            # Extract expression
            expr = query.lower().replace("calculate", "").strip()
            result = tools["calculator"](expr)
            return f"Result: {expr} = {result}"

        elif "weather" in query.lower():
            # Extract city
            city = "Tokyo"  # Simplified
            result = tools["weather"](city)
            return f"{result}. Recommendation: Bring an umbrella! ☂️"

        elif "search" in query.lower() or "find" in query.lower():
            result = tools["search"](query)
            return f"Found information: {result}"

        else:
            return "I'm not sure how to help with that."

    # Test the agent
    tasks = [
        "Calculate 25 + 37",
        "What's the weather today?",
        "Search for AI news",
    ]

    for task in tasks:
        print(agent(task))
        print()

simple_agent()

# ==============================================================================
# PART 4: MEMORY TYPES
# ==============================================================================

print("\n" + "=" * 60)
print("  AGENT MEMORY")
print("=" * 60)

print("""
   🧠 AGENTS HAVE MULTIPLE MEMORY TYPES:

   ┌─────────────────────────────────────────┐
   │  SHORT-TERM (Context Window)            │
   │  - Current conversation                 │
   │  - What's happening right now           │
   │  - Like: RAM                            │
   └─────────────────────────────────────────┘

   ┌─────────────────────────────────────────┐
   │  LONG-TERM (Vector Database)            │
   │  - Past conversations                   │
   │  - Retrieved when relevant              │
   │  - Like: Hard drive                     │
   └─────────────────────────────────────────┘

   Example:
   User: "Book a flight to Paris"
   Agent: Searches memories → Remembers user's preferences
          → Knows they prefer window seats + business class
""")

# ==============================================================================
# PART 5: AGENT FRAMEWORKS
# ==============================================================================

print("\n" + "=" * 60)
print("  POPULAR AGENT FRAMEWORKS")
print("=" * 60)

print("""
   🛠️  FRAMEWORKS = Pre-built tools for agents

   1️⃣ LANGGRAPH (LangChain)
   ─────────────────────────
   - Build complex workflows
   - Cycles and branches
   - State management
   - Production-ready

   2️⃣ CREWAI
   ───────────
   - Multi-agent systems
   - Assign roles (Researcher, Writer, etc.)
   - Collaborative tasks

   3️⃣ OPENAI AGENTS SDK
   ─────────────────────
   - Simple, elegant
   - Built by OpenAI
   - Easy handoffs between agents

   4️⃣ AUTOGEN (Microsoft)
   ───────────────────────
   - Conversation-based
   - Good for coding agents
   - Multi-agent chat

   5️⃣ SMOLAGENTS (HuggingFace)
   ─────────────────────────────
   - Lightweight, minimal
   - Open source
   - Code agents
""")

# ==============================================================================
# PART 6: REAL AGENT EXAMPLE (REACT PATTERN)
# ==============================================================================

def react_agent_example():
    """ReAct pattern: Reason + Act."""
    print("\n🎯 REACT AGENT EXAMPLE")
    print("-" * 40)

    print("""
    Task: "Should I bring a jacket in San Francisco today?"

    ──────────────────────────────────────

    Thought: The user wants clothing advice for San Francisco.
             I should check the weather first.

    Action: weather_tool(city="San Francisco")

    Observation: 15°C, windy, possible fog

    ──────────────────────────────────────

    Thought: It's quite cool and windy. A light jacket would be
             comfortable, especially in the evening.

    Action: None (I'm done reasoning)

    Final Answer: Yes, bring a light jacket! San Francisco is
                  15°C, windy, and can get chilly in the evening.
                  Layers work best.
    """)

react_agent_example()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
✅ AI Agent = LLM + Tools + Memory + Actions

✅ Agent Loop: Observe → Reason → Act → Observe → Repeat

✅ Tools extend capabilities (search, code, files, APIs)

✅ Memory: Short-term (context) + Long-term (vector store)

✅ Popular frameworks: LangGraph, CrewAI, AutoGen

✅ Agents can: Browse, code, calculate, email, search docs

NEXT: Build an agent with LangGraph!
""")