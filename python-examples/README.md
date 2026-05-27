# AI Engineering Masterclass - Python Examples

This directory contains Python code examples for each topic covered in the masterclass.

## Structure

```
python-examples/
├── Track1_Foundations/
│   ├── 1_1_math_basics.py          # Linear algebra, calculus, probability
│   ├── 1_2_python_numpy.py         # NumPy, Pandas, Matplotlib
│   ├── 1_3_data_engineering.py     # ETL, feature engineering, data splits
│   └── 1_4_visualizations.py       # Data visualization examples
├── Track2_Classical_ML/
│   ├── 2_1_linear_regression.py    # Linear & Logistic Regression
│   ├── 2_2_decision_trees.py       # Decision Trees & Random Forests
│   ├── 2_3_svm_kmeans.py          # SVMs, K-Means, DBSCAN
│   ├── 2_4_bias_variance.py        # Bias-Variance tradeoff experiments
│   └── 2_5_sklearn_pipeline.py    # Full Scikit-learn pipelines
├── Track3_Deep_Learning/
│   ├── 3_1_nn_from_scratch.py     # Neural network from scratch
│   ├── 3_2_cnn.py                 # Convolutional Neural Networks
│   ├── 3_3_rnn_lstm.py            # RNNs, LSTMs, GRUs
│   ├── 3_4_pytorch_pretrained.py  # Using pretrained models
│   └── 3_5_training_loop.py       # Custom training loops
├── Track4_Transformers/
│   ├── 4_1_attention.py           # Attention mechanism implementation
│   ├── 4_2_transformer.py          # Transformer architecture
│   ├── 4_3_tokenization.py        # Tokenization examples
│   └── 4_4_huggingface.py         # Hugging Face transformers
├── Track5_LLM_Ecosystem/
│   ├── 5_1_api_calls.py           # OpenAI, Anthropic API examples
│   ├── 5_2_ollama_local.py        # Using Ollama for local models
│   └── 5_3_multimodal.py          # Vision and audio models
├── Track6_Prompt_Engineering/
│   ├── 6_1_basic_prompts.py       # Zero-shot, few-shot examples
│   ├── 6_2_chain_of_thought.py    # CoT prompting
│   └── 6_3_structured_output.py    # JSON mode, function calling
├── Track7_RAG/
│   ├── 7_1_embeddings.py         # Creating embeddings
│   ├── 7_2_vector_db.py           # ChromaDB, Pinecone examples
│   ├── 7_3_rag_pipeline.py        # Complete RAG implementation
│   └── 7_4_advanced_rag.py        # HyDE, reranking, GraphRAG
├── Track8_Fine_Tuning/
│   ├── 8_1_lora.py                # LoRA fine-tuning with PEFT
│   ├── 8_2_qlora.py               # QLoRA for quantized training
│   ├── 8_3_dpo.py                 # DPO (Direct Preference Optimization)
│   └── 8_4_eval.py                # Model evaluation
├── Track9_Agents/
│   ├── 9_1_simple_agent.py        # Basic ReAct agent
│   ├── 9_2_langgraph.py           # LangGraph agent implementation
│   ├── 9_3_crewai.py              # Multi-agent crew
│   └── 9_4_tools.py               # Tool definition and calling
├── Track10_MLOps/
│   ├── 10_1_langchain.py          # LangChain chains and LCEL
│   ├── 10_2_evaluation.py          # LLM evaluation frameworks
│   └── 10_3_deployment.py          # vLLM, TGI deployment
├── Track11_Hardware/
│   └── 11_1_gpu_optimization.py   # GPU memory optimization examples
└── Track12_Frontier/
    ├── 12_1_mamba.py              # State Space Models
    └── 12_2_multimodal.py         # Multimodal examples
```

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn torch transformers
pip install openai anthropic langchain langgraph crewai
pip install chromadb pinecone-client sentence-transformers
pip install peft bitsandbytes huggingface-hub
```

## Usage

Each file is self-contained and includes:
1. Educational comments explaining the concept
2. Working code that can be run immediately
3. Visualization or output examples
4. References to related concepts

Run any file with:
```bash
python <file_path>
```