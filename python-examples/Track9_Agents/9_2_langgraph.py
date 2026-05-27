"""
Track 9.2: Building Agents with LangGraph
==========================================
Learn to build stateful, cyclical AI workflows using LangGraph.

Key concepts:
- StateGraph: Define nodes (agents/tools) and edges
- Conditional edges: Route based on state
- Persistence: Checkpoint and resume workflows
- Human-in-the-loop: Pause for approval

Author: AI Engineering Masterclass
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator

# ==============================================================================
# PART 1: BASIC CONCEPTS
# ==============================================================================

print("=" * 70)
print("  LANGGRAPH FUNDAMENTALS")
print("=" * 70)

# Define the state schema
class AgentState(TypedDict):
    """State managed by the graph."""
    messages: Annotated[list, operator.add]
    step: int
    decision: str

# ==============================================================================
# PART 2: SIMPLE ROUTING AGENT
# ==============================================================================

def create_router_agent():
    """
    Simple agent that classifies and routes to specialized handlers.
    """

    print("\n" + "-" * 50)
    print("EXAMPLE 1: Router Agent")
    print("-" * 50)

    # Define nodes as functions
    def classifier(state: AgentState) -> AgentState:
        """Classify the user's request."""
        messages = state["messages"]
        last_message = messages[-1]["content"]

        # Simple keyword-based routing
        if any(word in last_message.lower() for word in ["search", "find", "look up"]):
            decision = "research"
        elif any(word in last_message.lower() for word in ["write", "create", "draft"]):
            decision = "writer"
        elif any(word in last_message.lower() for word in ["code", "program", "function"]):
            decision = "coder"
        else:
            decision = "general"

        return {"step": state["step"] + 1, "decision": decision, "messages": []}

    def research_handler(state: AgentState) -> AgentState:
        """Handle research requests."""
        return {
            "step": state["step"] + 1,
            "messages": [{"role": "assistant", "content": "I'll search for that information..."}]
        }

    def writer_handler(state: AgentState) -> AgentState:
        """Handle writing requests."""
        return {
            "step": state["step"] + 1,
            "messages": [{"role": "assistant", "content": "I'll write that content for you..."}]
        }

    def coder_handler(state: AgentState) -> AgentState:
        """Handle coding requests."""
        return {
            "step": state["step"] + 1,
            "messages": [{"role": "assistant", "content": "I'll help you with that code..."}]
        }

    def general_handler(state: AgentState) -> AgentState:
        """Handle general requests."""
        return {
            "step": state["step"] + 1,
            "messages": [{"role": "assistant", "content": "I'll try to help with your request..."}]
        }

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classifier", classifier)
    workflow.add_node("researcher", research_handler)
    workflow.add_node("writer", writer_handler)
    workflow.add_node("coder", coder_handler)
    workflow.add_node("general", general_handler)

    # Set entry point
    workflow.set_entry_point("classifier")

    # Conditional routing based on classifier decision
    def route_based_on_decision(state: AgentState) -> str:
        """Determine next node based on classification."""
        return state["decision"]

    workflow.add_conditional_edges(
        "classifier",
        route_based_on_decision,
        {
            "research": "researcher",
            "writer": "writer",
            "coder": "coder",
            "general": "general"
        }
    )

    # All handlers end at END node
    for handler in ["researcher", "writer", "coder", "general"]:
        workflow.add_edge(handler, END)

    # Compile
    app = workflow.compile()

    # Run examples
    test_messages = [
        {"role": "user", "content": "Search for the latest AI research papers"},
        {"role": "user", "content": "Write me a poem about coding"},
        {"role": "user", "content": "Help me write a Python function"},
        {"role": "user", "content": "What is the weather like?"},
    ]

    for msg in test_messages:
        state = {"messages": [msg], "step": 0, "decision": ""}
        result = app.invoke(state)
        print(f"\nInput: {msg['content']}")
        print(f"Route: {result.get('decision', 'N/A')}")
        print(f"Response: {result['messages'][-1]['content']}")

# ==============================================================================
# PART 3: REACT AGENT (Reasoning + Acting)
# ==============================================================================

def create_react_agent():
    """
    ReAct agent that interleaves reasoning and tool execution.
    """

    print("\n" + "-" * 50)
    print("EXAMPLE 2: ReAct Agent (Reasoning + Acting)")
    print("-" * 50)

    class Thought:
        """Track thought process."""
        def __init__(self):
            self.steps = []

        def think(self, thought_text: str):
            self.steps.append(("thought", thought_text))
            print(f"🤔 Thinking: {thought_text}")

        def act(self, action: str, observation: str = ""):
            self.steps.append(("action", action))
            print(f"🔧 Acting: {action}")
            if observation:
                print(f"👁️ Observed: {observation}")
                self.steps.append(("observation", observation))

        def get_trajectory(self):
            return self.steps

    # Simulated tools
    def search_web(query: str) -> str:
        """Simulate web search."""
        return f"Results for '{query}': Article 1, Article 2, Article 3"

    def calculator(expr: str) -> str:
        """Simulate calculator."""
        try:
            result = eval(expr)
            return str(result)
        except:
            return "Error in calculation"

    def get_weather(location: str) -> str:
        """Simulate weather API."""
        return f"Weather in {location}: 72°F, Partly Cloudy"

    # Tool registry
    tools = {
        "search": search_web,
        "calculator": calculator,
        "weather": get_weather
    }

    def react_agent(question: str, max_iterations: int = 5):
        """Execute ReAct loop."""

        thought = Thought()
        observation = ""

        for i in range(max_iterations):
            # Thought
            if i == 0:
                thought.think(f"Question: {question}. I need to determine how to answer this.")
            else:
                thought.think(f"Based on what I observed, I need to continue my reasoning.")

            # Decide action
            if "search" in question.lower() or "find" in question.lower():
                action = "search"
                params = question
                observation = tools["search"](params)
                thought.act(f"Calling search('{params}')", observation)

            elif "calculate" in question.lower() or any(c in question for c in "+-*/"):
                thought.act("Need to calculate the expression")
                # Extract expression (simple demo)
                import re
                expr = re.findall(r'[\d\+\-*/().]+', question)
                if expr:
                    observation = tools["calculator"](expr[0])
                    thought.act(f"Calling calculator('{expr[0]}')", observation)

            elif "weather" in question.lower():
                # Extract location
                words = question.lower().split()
                loc_idx = words.index("weather") + 2 if "in" in words else len(words)
                location = " ".join(words[loc_idx:loc_idx+2]) if loc_idx < len(words) else "New York"
                observation = tools["weather"](location)
                thought.act(f"Calling weather('{location}')", observation)

            else:
                thought.think("This question doesn't require tool use")
                thought.act("Providing final answer based on reasoning")
                observation = f"Final answer to '{question}': Based on my analysis..."
                thought.act("Final response", observation)
                break

            # Check if done
            thought.think("Do I have enough information to answer?")
            if observation and "search" not in question.lower():
                break

        return thought.get_trajectory()

    # Run ReAct agent
    print("\nReAct Agent Execution:")
    print("=" * 50)

    questions = [
        "What's the weather in San Francisco?",
        "Calculate 15 * 23 + 45",
    ]

    for q in questions:
        print(f"\n❓ Question: {q}")
        trajectory = react_agent(q)
        print()

# ==============================================================================
# PART 4: RESEARCH AGENT WITH LANGGRAPH
# ==============================================================================

def create_research_agent():
    """
    Production-grade research agent with web search, note-taking, and synthesis.
    """

    print("\n" + "-" * 50)
    print("EXAMPLE 3: LangGraph Research Agent")
    print("-" * 50)

    # Define agent state
    class ResearchState(TypedDict):
        query: str
        search_results: Annotated[list, operator.add]
        notes: Annotated[list, operator.add]
        outline: str
        draft: str
        revision: str
        status: str

    # Simulated search function
    def search_web(state: ResearchState) -> ResearchState:
        """Simulate web search."""
        query = state["query"]
        results = [
            {"title": f"Article about {query}", "snippet": f"Key findings on {query}..."},
            {"title": f"{query} research paper", "snippet": "Academic research summary..."},
            {"title": f"Practical guide to {query}", "snippet": "How-to guide with steps..."},
        ]
        return {"search_results": results, "status": "searched"}

    def take_notes(state: ResearchState) -> ResearchState:
        """Extract key points from search results."""
        notes = []
        for i, result in enumerate(state.get("search_results", [])):
            notes.append(f"Note {i+1}: {result['title']} - {result['snippet'][:100]}")
        return {"notes": notes, "status": "notes_taken"}

    def create_outline(state: ResearchState) -> ResearchState:
        """Create document outline."""
        notes = state.get("notes", [])
        outline = f"# Research Report: {state['query']}\n\n"
        outline += "## Introduction\n"
        outline += "## Key Findings\n"
        for note in notes[:3]:
            outline += f"- {note}\n"
        outline += "## Conclusion"
        return {"outline": outline, "status": "outlined"}

    def write_draft(state: ResearchState) -> ResearchState:
        """Write initial draft."""
        draft = state.get("outline", "# Draft\n") + "\n\n[Full article content would go here...]"
        return {"draft": draft, "status": "drafted"}

    def revise_draft(state: ResearchState) -> ResearchState:
        """Review and improve draft."""
        return {"revision": state["draft"], "status": "completed"}

    # Build graph
    workflow = StateGraph(ResearchState)

    workflow.add_node("search", search_web)
    workflow.add_node("notes", take_notes)
    workflow.add_node("outline", create_outline)
    workflow.add_node("draft", write_draft)
    workflow.add_node("revise", revise_draft)

    workflow.set_entry_point("search")
    workflow.add_edge("search", "notes")
    workflow.add_edge("notes", "outline")
    workflow.add_edge("outline", "draft")
    workflow.add_edge("draft", "revise")
    workflow.add_edge("revise", END)

    app = workflow.compile()

    # Run research
    initial_state = {
        "query": "Latest developments in LLM fine-tuning",
        "search_results": [],
        "notes": [],
        "outline": "",
        "draft": "",
        "revision": "",
        "status": "init"
    }

    print("\nRunning Research Agent...")
    for state in app.stream(initial_state):
        print(f"  Node completed: {list(state.keys())}")

    print("\n✅ Research complete!")
    print(f"Outline:\n{state.get('revise', state.get('draft', '').split('[')[0])}")

# ==============================================================================
# PART 5: MULTI-AGENT SYSTEM
# ==============================================================================

def create_multi_agent_system():
    """
    Orchestrator + specialists pattern for complex tasks.
    """

    print("\n" + "-" * 50)
    print("EXAMPLE 4: Multi-Agent Research Team")
    print("-" * 50)

    class TeamState(TypedDict):
        task: str
        researchers_done: Annotated[int, operator.count]
        writer_done: bool
        critic_done: bool
        research_findings: list
        draft: str
        critiques: list
        final_report: str

    # Specialist agents
    def researcher_1(state: TeamState) -> TeamState:
        """Technical researcher."""
        return {
            "research_findings": state.get("research_findings", []) + ["Technical aspect 1 found"],
            "researchers_done": 1  # count increment
        }

    def researcher_2(state: TeamState) -> TeamState:
        """Market researcher."""
        return {
            "research_findings": state.get("research_findings", []) + ["Market trend discovered"],
            "researchers_done": 1
        }

    def writer(state: TeamState) -> TeamState:
        """Content writer."""
        findings = state.get("research_findings", [])
        draft = "# Research Report\n\n" + "\n\n".join(findings)
        return {"draft": draft, "writer_done": True}

    def critic(state: TeamState) -> TeamState:
        """Quality assurance critic."""
        return {"critiques": ["Consider adding more specificity", "Check facts"], "critic_done": True}

    def synthesizer(state: TeamState) -> TeamState:
        """Finalize report."""
        return {"final_report": state["draft"] + "\n\n--- Reviewed by critic ---"}

    # Build workflow
    workflow = StateGraph(TeamState)

    workflow.add_node("tech_researcher", researcher_1)
    workflow.add_node("market_researcher", researcher_2)
    workflow.add_node("writer", writer)
    workflow.add_node("critic", critic)
    workflow.add_node("synthesizer", synthesizer)

    workflow.set_entry_point("tech_researcher")

    # Parallel researchers
    workflow.add_edge("tech_researcher", "market_researcher")
    workflow.add_edge("market_researcher", "writer")

    # Writer and critic run in parallel
    def after_writing(state: TeamState) -> str:
        if state.get("critic_done"):
            return "synthesizer"
        return "critic"

    workflow.add_conditional_edges(
        "writer",
        lambda s: "synthesizer" if s.get("critic_done") else "critic",
        {"critic": "critic", "synthesizer": "synthesizer"}
    )
    workflow.add_edge("critic", "writer")  # Reviews go back to writer
    workflow.add_edge("synthesizer", END)

    app = workflow.compile()

    # Run team
    initial_state = {
        "task": "Research AI market trends",
        "researchers_done": 0,
        "writer_done": False,
        "critic_done": False,
        "research_findings": [],
        "draft": "",
        "critiques": [],
        "final_report": ""
    }

    print("\nRunning Multi-Agent Team...")
    results = app.invoke(initial_state)
    print("\n✅ Team complete!")
    print(f"Final report:\n{results['final_report']}")

# ==============================================================================
# PART 6: LOADING SAVING STATE
# ==============================================================================

def persistence_demo():
    """
    Demonstrate checkpointing (persistence) in LangGraph.
    Save and resume workflow state.
    """

    print("\n" + "-" * 50)
    print("EXAMPLE 5: Persistence & Human-in-the-Loop")
    print("-" * 50)

    print("""
    LangGraph provides built-in persistence via checkpointers:

    ```python
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Create checkpointer
    checkpointer = SqliteSaver.from_conn_string(":memory:")

    # Compile with checkpointer
    app = workflow.compile(checkpointer=checkpointer)

    # Run with thread_id (allows resumption)
    config = {"configurable": {"thread_id": "session_123"}}
    app.invoke(input_state, config)

    # Later: resume from where you left off
    app.invoke(None, config)  # None input = resume
    ```

    Human-in-the-Loop Pattern:

    ```python
    def should_continue(state):
        if len(state["citations"]) < 3:
            return "researcher"
        return "human_approval"

    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {"human_approval": "human_node"}
    )

    def human_node(state):
        # Pauses for human input!
        response = input("Approve this citation? y/n: ")
        return {"approved": response == "y"}
    ```
    """)

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  LANGGRAPH AGENTS - FULL WALKTHROUGH")
    print("=" * 70)

    # Run examples
    create_router_agent()
    create_react_agent()
    create_research_agent()
    create_multi_agent_system()
    persistence_demo()

    print("\n" + "=" * 70)
    print("  KEY LANGGRAPH PATTERNS SUMMARY")
    print("=" * 70)
    print("""
    1. STATEGRpah: Foundation for all workflows
       - Define state schema (TypedDict)
       - Add nodes (functions)
       - Define edges (sequential or conditional)

    2. CONDITIONAL ROUTING:
       - Route to different nodes based on state
       - Enables specialized handling

    3. PERSISTENCE:
       - Checkpoint state to resume later
       - Enable multi-session support

    4. HUMAN-IN-THE-LOOP:
       - Interrupt workflow for approval
       - Resume with human input

    5. CYCLICAL WORKFLOWS:
       - Loop back to earlier nodes
       - Implement ReAct-style reasoning

    NEXT: Build a production agent with LangGraph
    """)

# ==============================================================================
# PRODUCTION EXAMPLE (Commented for reference)
# ==============================================================================

PRODUCTION_CODE = '''
"""
Production Research Agent (Reference Implementation)
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# Tools
search = DuckDuckGoSearchRun()
tool_executor = ToolExecutor([search])

# State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    outcome: str

# LLM with tools
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools([search])

# Nodes
def research_agent(state: AgentState):
    """Call LLM with tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def execute_tools(state: AgentState):
    """Execute tool calls from LLM."""
    last_message = state["messages"][-1]
    tool_calls = last_message.additional_kwargs.get("tool_calls", [])

    results = []
    for call in tool_calls:
        result = tool_executor.invoke(call)
        results.append({"tool_call_id": call["id"], "content": result})

    return {"messages": results}

def should_act(state: AgentState):
    """Route based on whether we need to act."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "additional_kwargs"):
        if last_message.additional_kwargs.get("tool_calls"):
            return "tool_executor"
    return END

# Build
workflow = StateGraph(AgentState)
workflow.add_node("agent", research_agent)
workflow.add_node("tool_executor", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_act, {
    "tool_executor": "tool_executor",
    END: END
})
workflow.add_edge("tool_executor", "agent")

app = workflow.compile()

# Run
initial_state = {
    "messages": [HumanMessage(content="Search for recent AI breakthroughs")]
}
for chunk in app.stream(initial_state, stream_mode="updates"):
    print(chunk)
'''

print("\n📝 Production code template available in source file.")
