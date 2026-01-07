"""
Performance and threading controls for faster training/inference.

- Limits BLAS/OMP thread oversubscription
- Disables HF tokenizers parallelism warnings/threads
- Sets PyTorch intra/inter-op thread counts
"""

import os
import multiprocessing as mp


def apply_optimizations(threads: int | None = None, interop_threads: int | None = None):
    # Decide defaults based on CPU count
    cpu_count = max(1, mp.cpu_count())
    if threads is None:
        # Keep a few cores free for OS; cap to 4 to avoid oversubscription on laptops
        threads = min(4, cpu_count)
    if interop_threads is None:
        interop_threads = 1

    # Limit math libraries threads to prevent oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))

    # HuggingFace tokenizers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch

        # Set PyTorch threads
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(interop_threads)
        # Enable matmul TF32 on Ampere+ GPUs for speed (safe for DL inference/training)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        # Torch may not be installed yet
        pass
