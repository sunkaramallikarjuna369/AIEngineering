# AI Engineering Masterclass - Python Examples

Beginner-friendly, well-commented Python examples for learning AI Engineering from scratch!

## 🎓 Learning Path

Start with **BEGINNER** files, then move to advanced ones.

---

## 📚 Beginner-Friendly Files (Start Here!)

| Track | File | What You'll Learn |
|-------|------|------------------|
| 1 | [1_2_python_basics.py](Track1_Foundations/1_2_python_basics.py) | NumPy, Pandas, Matplotlib - the basics |
| 1 | [1_3_data_engineering.py](Track1_Foundations/1_3_data_engineering.py) | Data cleaning, feature engineering |
| 2 | [2_1_ml_basics.py](Track2_Classical_ML/2_1_ml_basics.py) | What is ML? Types of ML? |
| 3 | [3_1_nn_simple.py](Track3_Deep_Learning/3_1_nn_simple.py) | Neural networks from scratch |
| 4 | [4_1_attention_simple.py](Track4_Transformers/4_1_attention_simple.py) | Attention mechanism explained simply |
| 5 | [5_1_llm_basics.py](Track5_LLM_Ecosystem/5_1_llm_basics.py) | What are LLMs? How do they work? |
| 7 | [7_1_rag_basics.py](Track7_RAG/7_1_rag_basics.py) | RAG explained - give AI your docs |
| 9 | [9_1_agent_basics.py](Track9_Agents/9_1_agent_basics.py) | AI Agents - tools, memory, actions |

---

## 📁 Complete File Structure

```
python-examples/
├── Track1_Foundations/
│   ├── 1_1_math_basics.py              # Linear algebra, calculus, probability
│   ├── 1_2_python_basics.py            # ⭐ BEGINNER: Python essentials
│   └── 1_3_data_engineering.py         # ⭐ BEGINNER: Data prep & cleaning
│
├── Track2_Classical_ML/
│   ├── 2_1_ml_basics.py                # ⭐ BEGINNER: ML fundamentals
│   └── 2_5_sklearn_pipeline.py         # Full ML pipelines with sklearn
│
├── Track3_Deep_Learning/
│   ├── 3_1_nn_from_scratch.py          # Neural networks (advanced)
│   └── 3_1_nn_simple.py                # ⭐ BEGINNER: Neural nets explained
│
├── Track4_Transformers/
│   ├── 4_1_attention.py                # Attention mechanism (advanced)
│   └── 4_1_attention_simple.py         # ⭐ BEGINNER: Attention made easy
│
├── Track5_LLM_Ecosystem/
│   ├── 5_1_api_calls.py               # OpenAI, Claude, Gemini APIs
│   └── 5_1_llm_basics.py              # ⭐ BEGINNER: LLM fundamentals
│
├── Track6_Prompt_Engineering/
│   └── 6_1_prompt_patterns.py         # Prompt techniques & patterns
│
├── Track7_RAG/
│   ├── 7_1_rag_basics.py              # ⭐ BEGINNER: RAG explained
│   └── 7_3_rag_pipeline.py            # Full RAG implementation
│
├── Track8_Fine_Tuning/
│   └── 8_1_lora.py                    # LoRA, QLoRA fine-tuning
│
├── Track9_Agents/
│   ├── 9_1_agent_basics.py           # ⭐ BEGINNER: AI Agents explained
│   └── 9_2_langgraph.py               # LangGraph agent implementation
│
├── Track10_MLOps/
│   └── 10_1_deployment.py              # vLLM, deployment, monitoring
│
├── Track11_Hardware/
│   └── 11_1_gpu_optimization.py        # GPU optimization
│
└── Track12_Frontier/
    └── 12_1_frontier.py               # Emerging tech, Mamba, safety
```

---

## 🚀 Quick Start

### For Beginners (Start Here!)
```bash
# 1. Learn Python basics
python python-examples/Track1_Foundations/1_2_python_basics.py

# 2. Understand ML
python python-examples/Track2_Classical_ML/2_1_ml_basics.py

# 3. Neural networks made simple
python python-examples/Track3_Deep_Learning/3_1_nn_simple.py

# 4. How attention works
python python-examples/Track4_Transformers/4_1_attention_simple.py

# 5. What are LLMs?
python python-examples/Track5_LLM_Ecosystem/5_1_llm_basics.py

# 6. RAG explained
python python-examples/Track7_RAG/7_1_rag_basics.py

# 7. AI Agents
python python-examples/Track9_Agents/9_1_agent_basics.py
```

### Run Any Example
```bash
# From project root
python python-examples/Track1_Foundations/1_2_python_basics.py

# Or from python-examples folder
cd python-examples
python Track1_Foundations/1_2_python_basics.py
```

---

## 📦 Requirements

```bash
# Core libraries
pip install numpy pandas matplotlib scikit-learn

# Deep Learning
pip install torch torchvision

# LLMs & Agents
pip install openai anthropic langchain langgraph

# Embeddings & RAG
pip install sentence-transformers chromadb

# Fine-tuning
pip install peft bitsandbytes transformers
```

---

## 🎯 Learning Roadmap

```
Week 1-2: Python & Data
├── 1_2_python_basics.py      → Learn NumPy, Pandas, Matplotlib
├── 1_3_data_engineering.py   → Data cleaning & feature engineering
└── 2_1_ml_basics.py          → ML fundamentals

Week 3-4: Deep Learning
├── 3_1_nn_simple.py          → Neural networks (intuition)
├── 3_1_nn_from_scratch.py     → Build NN from scratch
└── 4_1_attention_simple.py    → Attention mechanism

Week 5-6: LLMs & RAG
├── 5_1_llm_basics.py         → What are LLMs?
├── 5_1_api_calls.py          → Using LLM APIs
├── 7_1_rag_basics.py         → RAG explained
└── 7_3_rag_pipeline.py       → Build RAG pipeline

Week 7-8: Agents & Production
├── 9_1_agent_basics.py       → AI Agents explained
├── 9_2_langgraph.py          → Build agents
├── 8_1_lora.py               → Fine-tuning
└── 10_1_deployment.py        → Deployment
```

---

## 💡 Tips for Beginners

1. **Read the comments** - Each line is explained
2. **Run the code** - Don't just read, execute!
3. **Experiment** - Change values and see what happens
4. **Ask questions** - If something is unclear, ask!

---

## 📖 Each File Contains

✅ Clear explanations (no jargon without explanation)
✅ Real-world analogies
✅ Working code you can run
✅ Visual examples
✅ "What to remember" summary at the end
✅ Links to next concepts

---

## 🔗 Related Resources

- 📺 Andrej Karpathy's YouTube (Neural networks)
- 📚 Fast.ai Courses (Practical deep learning)
- 📖 Hands-On Machine Learning (Géron)
- 🛠️ Hugging Face Courses (Transformers)

---

*Happy Learning! 🚀*
