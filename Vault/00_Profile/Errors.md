---
type: profile
tags: [profile, error-history, gotchas]
last_updated: 2026-10-22
---
# Personal Deep Learning Error History & Engineering Gotchas

This file tracks recurring gotchas, GPU bottlenecks, and debugging lessons.
Entries are retrieved dynamically to avoid repeating past mistakes.

- **[CUDA_OOM_TENSOR]**:
  - *Symptom*: `torch.cuda.OutOfMemoryError: CUDA out of memory.`
  - *Context*: RTX 4050 Laptop GPU (6141 MiB limit). Batch size > 32 on Vision Transformers or feature maps with high resolution ($>512\times512$).
  - *Verified Fix*: Lower batch size to 16 or 8, enable `torch.cuda.amp.autocast(dtype=torch.float16)`, and accumulate gradients over 2 or 4 steps.

- **[LOSS_NAN_AMP]**:
  - *Symptom*: Training loss turns to `nan` or `inf` after 3-5 epochs.
  - *Context*: Deep residual or transformer networks using standard AdamW and FP16 automatic mixed precision without gradient scaling.
  - *Verified Fix*: Always use `torch.cuda.amp.GradScaler()` with AMP and clip gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

- **[CHROMADB_SIGNATURE_STRICT]**:
  - *Symptom*: `ValueError: Expected EmbeddingFunction.__call__ to have signature ['self', 'input']`.
  - *Context*: ChromaDB $\ge 0.4.16$ inspects `__call__` signature strictly.
  - *Verified Fix*: Define `def __call__(self, input: List[str]) -> List[List[float]]:` without `**kwargs` and implement `embed_query(self, input: List[str]) -> List[List[float]]`.

- **[WINDOWS_CP1254_UNICODE]**:
  - *Symptom*: `UnicodeEncodeError: 'charmap' codec can't encode character in position...`
  - *Context*: Turkish Windows default code page `cp1254` crashing when writing emojis or Unicode mathematical symbols ($\checkmark, \times, \bullet$) to console stdout.
  - *Verified Fix*: Use ASCII indicators (`[OK]`, `[FAIL]`, `*`) or rewrap stdout with `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`.

- **[CONV2D_CHANNEL_MISMATCH]**:
  - *Symptom*: `RuntimeError: Given groups=1, weight of size [C_out, C_in, K, K], expected input to have C_in channels, but got C_other channels instead`.
  - *Context*: Connecting output of dimension reduction (e.g. $1\times1$ conv with reduction ratio $r$) back into channel attention or residual addition.
  - *Verified Fix*: Verify $C/r$ is restored back to $C$ using an expansion $1\times1$ convolution before computing sigmoid attention weights or residual add.
