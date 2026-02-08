# LLM Training Performance: Profiling & Optimization on NVIDIA Grace Hopper

Summary of techniques from NVIDIA’s Grace Hopper–focused posts on **profiling** LLM training workflows and **advanced optimization** strategies. These methods address hardware limits and help scale LLM training.

**Sources:**

- [Profiling LLM Training Workflows on NVIDIA Grace Hopper](https://developer.nvidia.com/blog/profiling-llm-training-workflows-on-nvidia-grace-hopper/)
- [Advanced Optimization Strategies for LLM Training on NVIDIA Grace Hopper](https://developer.nvidia.com/blog/advanced-optimization-strategies-for-llm-training-on-nvidia-grace-hopper)

---

## Hardware Context: Grace Hopper Superchip

| Component | Description |
|-----------|-------------|
| **CPU** | Grace: 72 Arm Neoverse V2 cores, up to 480 GB LPDDR5X, ~500 GB/s bandwidth |
| **GPU** | Hopper: 96 GB or 144 GB HBM3/HBM3e, up to ~4.9 TB/s bandwidth |
| **Link** | **NVLink-C2C** at ~900 GB/s between CPU and GPU (much faster than PCIe) |

The tight CPU–GPU coupling is what makes CPU offloading and unified memory effective on this platform.

---

## 1. Profiling Techniques

Use these to understand where time and resources are spent before optimizing.

| Tool / technique | Purpose |
|------------------|--------|
| **NVIDIA Nsight Systems** | Application-level profiling and metric sampling for the full workflow |
| **CUDA Profiler API** | Mark regions to profile and shrink trace files |
| **PyTorch Profiler** | Break down time per op/kernel in PyTorch training |
| **Selective iteration profiling** | Use env vars (e.g. `TLLM_PROFILE_START_STOP`) to profile only chosen steps and control profile size |
| **NVTX markers** | Annotate code for timelines; useful for debugging and GC behavior |
| **Chrome tracing** | Inspect and share profiles in a timeline view |

---

## 2. Advanced Optimization Techniques

These are the main levers for improving LLM training performance and scalability on Grace Hopper.

### CPU offloading

- Move part of the workload (e.g. optimizer states, some activations) to the **Grace CPU** so the GPU is not memory-bound.
- **SuperOffload** (PyTorch) is built for Grace–Hopper and can deliver:
  - Full fine-tuning of large models on a **single GH200** (e.g. GPT-OSS-20B, Qwen3-14B; reported up to ~600 TFLOPS).
  - Scaling to **Llama-70B** on four GH200s.
  - Up to **~4× higher throughput** than prior ZeRO-Offload–style solutions.
  - **GPU utilization** going from ~50% to well above 80%.
- Ideas inside SuperOffload: **Speculation-then-Validation (STV)**, **heterogeneous optimizer** (split across CPU/GPU), **superchip-aware casting** to place work on CPU vs GPU wisely.
- Integrates with **DeepSpeed ZeRO Stage 3** and **Hugging Face Transformers** without changing model code.

### Unified Memory & CPU–GPU memory sharing

- **Unified Memory** and **KV cache offloading** use a shared CPU–GPU address space so the GPU can use CPU memory when needed.
- Reduces GPU OOMs and allows running large models (e.g. **Llama 3 70B**) by spilling KV cache or other tensors to CPU over the fast NVLink-C2C link.

### Automatic Mixed Precision (AMP)

- **AMP** keeps most training in **FP16/BF16** and uses FP32 only where needed (e.g. loss scaling, sensitive ops).
- Lowers **memory use** and increases **throughput** while keeping training stable; commonly used with a **GradScaler** for loss scaling.

### FP8 training

- **FP8** (e.g. E4M3 / E5M2) further cuts memory and can boost throughput on Hopper (native FP8 support).
- Used for weights, activations, or both, with scaling to preserve range; best when combined with AMP and proper scaling.

---

## 3. Summary Table

| Technique | What it addresses | Main benefit |
|-----------|-------------------|--------------|
| **Profiling (Nsight, PyTorch Profiler, NVTX)** | Unknown bottlenecks | Find where to optimize |
| **CPU offloading (e.g. SuperOffload)** | GPU memory & underutilization | Fit larger models, higher GPU utilization |
| **Unified Memory / KV offload** | GPU memory limit | Run very large LLMs (e.g. 70B) with CPU–GPU sharing |
| **AMP (FP16/BF16)** | Memory & compute | Less memory, higher throughput, stable training |
| **FP8 training** | Memory & compute | Even lower memory and higher throughput on Hopper |

---

## 4. Takeaway

Advanced optimizations—**CPU offloading**, **Unified Memory**, **AMP**, and **FP8**—together address both memory and compute limits. They are especially effective on Grace Hopper thanks to NVLink-C2C, and they enable training and fine-tuning larger LLMs than would fit with GPU-only, FP32 setups.

For more detail and exact commands, see the two NVIDIA blog posts linked at the top.
