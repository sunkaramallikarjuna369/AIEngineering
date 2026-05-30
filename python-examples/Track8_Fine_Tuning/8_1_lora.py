"""
Track 8.1: LoRA Fine-Tuning
===========================
Parameter-efficient fine-tuning with LoRA and QLoRA.

Author: AI Engineering Masterclass
"""

import torch
from typing import List, Dict

# ==============================================================================
# PART 1: LORA CONCEPT
# ==============================================================================

def lora_concept():
    """
    LoRA (Low-Rank Adaptation)原理:

    Instead of updating all weights W → W', LoRA adds a low-rank update:
    W' = W + ΔW = W + BA

    where:
    - W: frozen pre-trained weights (d × d)
    - B: trainable matrix (d × r)
    - A: trainable matrix (r × d)
    - r: rank (typically4-64)

    Total trainable params: 2dr vs d² for full fine-tuning
    For d=4096, r=16: 131K vs 16.7M parameters! 💡
    """

    print("LoRA reduces trainable parameters by ~100x!")
    print("Instead of W → W', we add BA where B(d×r), A(r×d)")

def lora_math():
    """The mathematics behind LoRA."""
    # Traditional fine-tuning: W' = W + ΔW
    # LoRA: W' = W + BA where rank(BA) = r << d

    # Example: d=768, r=8
    d = 768
    r = 8

    # Full fine-tuning: d² parameters
    full_params = d * d

    # LoRA: 2dr parameters
    lora_params = 2 * d * r

    reduction = full_params / lora_params

    print(f"Original weight matrix: {d}×{d} = {full_params:,} params")
    print(f"LoRA decomposition: 2×{d}×{r} = {lora_params:,} params")
    print(f"Parameter reduction: {reduction:.1f}x smaller!")

# ==============================================================================
# PART 2: PEFT LIBRARY
# ==============================================================================

def peft_lora_example():
    """Using HuggingFace PEFT for LoRA fine-tuning."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM

        # 1. Load base model
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        # 2. Configure LoRA
        config = LoraConfig(
            r=16,                          # Rank
            lora_alpha=32,                 # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Which layers to adapt
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # 3. Apply LoRA to model
        model = get_peft_model(model, config)

        # 4. Check trainable parameters
        model.print_trainable_parameters()
        # Output:
        # trainable params: 983,040 || all params: 124,355,584 || trainable%: 0.79%

        return model

    except ImportError:
        print("Install PEFT: pip install peft")
        return None

# ==============================================================================
# PART 3: QLORA (QUANTIZED LORA)
# ==============================================================================

def qlora_example():
    """
    QLoRA = Quantization + LoRA

    Process:
    1. Quantize base model to 4-bit (NF4)
    2. Keep LoRA params in full precision
    3. Train only LoRA parameters
    4. Merge during inference

    Enables fitting 70B models on single GPU!
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import prepare_model_for_kbit_training, get_peft_model
        from bitsandbytes import BitsAndBytesConfig

        #1. Quantization config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # 2. Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto"
        )

        # 3. Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # 4. Apply LoRA
        from peft import LoraConfig
        config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, config)

        print("QLoRA model ready! Can train7B model on single24GB GPU.")

 except ImportError:
        print("Install: pip install bitsandbytes peft transformers")

# ==============================================================================
# PART 4: PRACTICAL TRAINING LOOP
# ==============================================================================

def training_loop_example():
    """Complete LoRA training loop."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from datasets import load_dataset

        # 1. Load model and tokenizer
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # 2. Apply LoRA
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, config)

        # 3. Prepare dataset
        dataset = load_dataset("yelp_review_full", split="train[:1000]")
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=128),
            batched=True
        )

        # 4. Training arguments
        training_args = TrainingArguments(
            output_dir="./lora_model",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch"
        )

        # 5. Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        )
        trainer.train()

        # 6. Save
        model.save_pretrained("./lora_model")

 except ImportError as e:
        print(f"Install required: pip install peft transformers datasets")
        print(f"Error: {e}")

# ==============================================================================
# PART 5: DPO (DIRECT PREFERENCE OPTIMIZATION)
# ==============================================================================

def dpo_example():
    """
    DPO - Direct Preference Optimization

    Instead of RLHF (reward model + PPO), DPO directly optimizes
    using preference pairs without a separate reward model.

    Loss: -log(σ(log_π(y_w) - log_π(y_l))

    where y_w = preferred response, y_l = dispreferred response
    """
    try:
        from trl import DPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 1. Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        ref_model = AutoModelForCausalLM.from_pretrained("gpt2")

        # 2. DPO Trainer
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            beta=0.1,  # KL penalty coefficient
            train_dataset=preference_dataset  # Needs 'prompt', 'chosen', 'rejected'
        )

        dpo_trainer.train()
        model.save_pretrained("./dpo_model")

    except ImportError:
        print("Install: pip install trl")

# ==============================================================================
# PART 6: MERGING LORA WEIGHTS
# ==============================================================================

def merge_and_infer():
    """Merge LoRA weights for inference."""
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 1. Load base model
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")

        # 2. Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, "./lora_model")

        # 3. Merge weights (optional - for faster inference)
        merged_model = model.merge_and_unload()

        # 4. Use for inference
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        outputs = merged_model.generate(**inputs, max_new_tokens=20)
        print(tokenizer.decode(outputs[0]))

    except ImportError:
        print("Install: pip install peft transformers")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TRACK 8.1: LORA FINE-TUNING")
    print("=" * 70)

    print("\n1. LoRA Concept:")
    lora_concept()

    print("\n2. LoRA Mathematics:")
    lora_math()

    print("\n3. PEFT LoRA Example:")
    peft_lora_example()

    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. LoRA adds low-rank matrices (A, B) instead of updating all weights
    2. W' = W + BA where rank(BA) = r << d
    3. Typical rank values: 4, 8, 16, 32, 64
    4. Target modules: q_proj, v_proj (attention layers)
    5. QLoRA =4-bit quantization + LoRA (fits 70B on one GPU!)

    Common Use Cases:
    - Customize Llama for domain (medical, legal, code)
    - Add specific behaviors (personality, response style)
    - Adapt to user preferences
    - Multi-task learning (different adapters for different tasks)
    """)

    print("\n" + "=" * 70)
    print("  NEXT: Advanced RAG Patterns (Track 7.5)")
    print("=" * 70)