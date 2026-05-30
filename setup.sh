#!/bin/bash
# ============================================================
# AI Engineering Masterclass - Setup Script
# ============================================================
# Usage:
#   ./setup.sh          # Install all dependencies
#   ./setup.sh beginner # Install only beginner packages
#   ./setup.sh llm      # Install LLM packages
# ============================================================

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║   AI Engineering Masterclass - Setup                ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Detect package manager
if command -v uv &> /dev/null; then
    PKG_MANAGER="uv"
    echo "📦 Using: uv (fast Python package manager)"
elif command -v pip &> /dev/null; then
    PKG_MANAGER="pip"
    echo "📦 Using: pip"
else
    echo "❌ Error: No package manager found. Install uv or pip."
    exit 1
fi

echo ""

# Function to install packages
install_packages() {
    local packages=("$@")
    if [ "$PKG_MANAGER" = "uv" ]; then
        uv pip install "${packages[@]}"
    else
        pip install "${packages[@]}"
    fi
}

# Parse arguments
MODE="${1:-all}"

case $MODE in
    beginner)
        echo "🎯 Installing BEGINNER packages only..."
        echo ""
        install_packages numpy pandas matplotlib scikit-learn
        echo ""
        echo "✅ Beginner packages installed!"
        echo ""
        echo "Next steps:"
        echo "  1. Activate: source .venv/Scripts/activate"
        echo "  2. Run: python python-examples/Track1_Foundations/1_2_python_basics.py"
        ;;

    llm)
        echo "🤖 Installing LLM packages..."
        echo ""
        install_packages openai anthropic langchain langchain-openai langgraph
        install_packages sentence-transformers chromadb
        echo ""
        echo "✅ LLM packages installed!"
        ;;

    all|*)
        echo "🚀 Installing ALL packages..."
        echo ""

        echo "📚 Installing core packages..."
        install_packages numpy pandas matplotlib scikit-learn

        echo ""
        echo "🧠 Installing PyTorch..."
        install_packages torch torchvision

        echo ""
        echo "🤖 Installing LLM packages..."
        install_packages openai anthropic
        install_packages langchain langchain-openai langchain-anthropic langgraph

        echo ""
        echo "📦 Installing RAG packages..."
        install_packages sentence-transformers chromadb langchain-community

        echo ""
        echo "🔧 Installing development tools..."
        install_packages python-dotenv tqdm

        echo ""
        echo "╔═══════════════════════════════════════════════════════╗"
        echo "║   ✅ ALL PACKAGES INSTALLED!                          ║"
        echo "╚═══════════════════════════════════════════════════════╝"
        echo ""
        echo "Next steps:"
        echo "  1. Activate virtual environment:"
        echo "     source .venv/Scripts/activate  (Linux/Mac)"
        echo "     .venv\\Scripts\\activate         (Windows)"
        echo ""
        echo "  2. Set up API keys (optional):"
        echo "     export OPENAI_API_KEY=your-key"
        echo "     export ANTHROPIC_API_KEY=your-key"
        echo ""
        echo "  3. Run your first example:"
        echo "     python python-examples/Track1_Foundations/1_2_python_basics.py"
        ;;
esac

echo ""
echo "📖 For more info, see README.md"
