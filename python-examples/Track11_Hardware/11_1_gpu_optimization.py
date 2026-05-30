"""
Track 11.1: GPU Optimization for AI
==================================
Maximizing GPU utilization for training and inference.

Author: AI Engineering Masterclass
"""

import torch

# ==============================================================================
# PART 1: CUDA BASICS
# ==============================================================================

def cuda_setup():
    """Check and configure CUDA."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def tensor_on_gpu():
    """Move tensors to GPU."""
    # Create tensor on CPU
    x = torch.randn(1000, 1000)

    # Move to GPU
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        print(f"Tensor on: {x_gpu.device}")

        # Operations on GPU
        y = x_gpu @ x_gpu.T  # Matrix multiplication
        print(f"Result shape: {y.shape}")

# ==============================================================================
# PART 2: OPTIMIZATION TECHNIQUES
# ==============================================================================

def gradient_checkpointing():
    """Save memory during training by checkpointing."""
    from torch.utils.checkpoint import checkpoint

    class CheckpointedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(512, 512) for _ in range(20)
            ])

        def forward(self, x):
            for layer in self.layers:
                # Checkpoint expensive layers
                x = checkpoint(layer, x, use_reentrant=False)
            return x

def mixed_precision():
    """Use FP16 for faster training."""
    from torch.cuda.amp import autocast, GradScaler

    model = torch.nn.Linear(512, 512).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()

    for data, target in dataloader:
        optimizer.zero_grad()

        with autocast():
            output = model(data.cuda())
            loss = loss_fn(output, target.cuda())

        # Scales loss, calls backward, unscales gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def_flash_attention():
    """Use Flash Attention for efficiency."""
    try:
        from flash_attn import flash_attn_func

        # Q, K, V tensors (batch, seq, heads, dim)
        # Standard attention
        # output = flash_attn_func(Q, K, V, causal=True)

        print("Flash Attention available!")
        print("Benefits: 2-4x faster, 5-20x memory efficient")

    except ImportError:
        print("Install: pip install flash-attn")

# ==============================================================================
# PART 3: BATCHING & DATALOADER
# ==============================================================================

def optimal_batching():
    """Optimize batch sizes for GPU."""
    # Formula for optimal batch size:
    # batch_size = (GPU_memory * 0.9) / (model_size * 2 * precision)

    gpu_memory_gb = 24  # e.g., A10G
    model_size_gb = 7  # 7B parameter model
    precision = 2  # FP16 = 2 bytes

    # Calculate
    optimal_batch = int((gpu_memory_gb * 0.9) / (model_size_gb * precision))
    print(f"Optimal batch size: {optimal_batch}")

    # Memory breakdown
    memory_items = {
        "Model weights": model_size_gb * precision,
        "Gradients": model_size_gb * precision,
        "Optimizer states": model_size_gb * precision * 4,  # Adam = 4 bytes
        "Activations (estimate)": 2 * optimal_batch  # Depends on sequence length
    }

    print("\nMemory allocation:")
    for item, size in memory_items.items():
        print(f"  {item}: {size:.1f} GB")

def custom_dataloader():
    """Optimized data loading."""
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,          # Parallel data loading
        pin_memory=True,       # Faster GPU transfer
        prefetch_factor=2,     # Prefetch batches
        persistent_workers=True  # Keep workers alive
    )

# ==============================================================================
# PART 4: DISTRIBUTED TRAINING
# ==============================================================================

def distributed_setup():
    """Setup for multi-GPU training."""
    import torch.distributed as dist

    def setup_distributed():
        # Initialize
        dist.init_process_group(backend="nccl")  # NVIDIA NCCL

        # Get rank
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Set device
        torch.cuda.set_device(rank)

        return rank, world_size

    def cleanup():
        dist.destroy_process_group()

def data_parallel():
    """DataParallel vs DistributedDataParallel."""
    # DataParallel (single machine, easier)
    model = torch.nn.Linear(512, 512)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # DistributedDataParallel (multi-machine, faster)
    # from torch.nn.parallel import DistributedDataParallel as DDP
    # model = DDP(model, device_ids=[rank])

def_fsdp():
    """Fully Sharded Data Parallelism for massive models."""
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision
        )

        # Shard model across GPUs
        model = FullyShardedDataParallel(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        )

    except ImportError:
        print("Requires PyTorch with distributed support")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TRACK 11.1: GPU OPTIMIZATION")
    print("=" * 70)

    print("\n1. CUDA Setup:")
    cuda_setup()

    print("\n2. Quantization Comparison:")
    from Track10_MLOps import quantization_comparison
    quantization_comparison()

    print("\n" + "=" * 70)
    print("  GPU OPTIMIZATION CHECKLIST")
    print("=" * 70)
    print("""
    ✓ Use mixed precision (FP16/BF16)
    ✓ Enable gradient checkpointing
    ✓ Use Flash Attention when possible
    ✓ Optimize batch size for GPU memory
    ✓ Use pin_memory for faster transfers
    ✓ Use prefetch_factor in DataLoader
    ✓ Use DistributedDataParallel for multi-GPU
    ✓ Use FSDP for models exceeding single GPU memory
    ✓ Profile with torch.profiler
    ✓ Monitor with nvidia-smi
    """)

    print("\n" + "=" * 70)
    print("  HARDWARE COMPARISON")
    print("=" * 70)
    print("""
    NVIDIA A100: 80GB HBM, 2TB/s bandwidth
    NVIDIA H100: 80GB HBM, 3.35TB/s bandwidth
    NVIDIA L40S: 48GB, cost-effective inference
    AMD MI300X: 128GB HBM, good for MoE

    For training: H100 (8x) in DGX systems
    For inference: A100 or L40S for cost efficiency
    """)