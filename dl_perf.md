# CUDA Graphs vs torch.compile: Solving the CPU Bottleneck in Deep Learning

Both **CUDA Graphs** and **torch.compile** are designed to solve the **CPU bottleneck** in deep learning—where the CPU is too slow to tell the GPU what to do. This note gives a concise comparison.

---

## The Problems They Solve

| Problem | Description |
|--------|-------------|
| **Kernel launch overhead** | In *eager mode*, the CPU must launch every kernel (addition, multiplication, etc.) one by one. If you have 1,000 tiny kernels, the CPU takes longer to "issue the command" than the GPU takes to "do the math." |
| **Python overhead** | The Python interpreter is slow. Evaluating loops and logic in Python between every kernel launch wastes precious milliseconds. |
| **Memory bandwidth (HBM) bottleneck** | Standard code moves data from GPU memory to the processor and back for every operation. **torch.compile** specifically addresses this. |

---

## Similarities

- **Goal**: Both aim to keep the GPU fully utilized (reducing "gaps" in the timeline).
- **Static nature**: Both perform best when input shapes (e.g. batch size, sequence length) remain constant.
- **Warm-up**: Both require a warm-up phase (recording the graph or compiling the kernels) before they provide a speedup.

---

## Key Differences

| Feature | CUDA Graphs | torch.compile |
|---------|-------------|---------------|
| **Method** | **Recording**: Like a "tape recorder," it saves the exact sequence of kernel launches and replays them. | **Transformation**: Uses a compiler (Triton) to rewrite your code and fuse operations together. |
| **Kernel fusion** | **No.** It just launches existing kernels faster. | **Yes.** It merges multiple kernels into one (e.g. Linear + ReLU) to save memory trips. |
| **Flexibility** | **Very rigid.** Requires fixed memory addresses and fixed shapes. | **Flexible.** Can handle dynamic shapes (to an extent) and complex Python logic. |
| **Implementation** | **Manual.** You must handle "static buffers" and recording logic yourself. | **Automatic.** Just one line: `model = torch.compile(model)`. |
| **Scope** | Driver/runtime level (hardware feature). | Compiler/software level (software feature). |

---

## Golden Rule

- **Use CUDA Graphs** if your kernels are already highly optimized (e.g. in a C++ backend) but launch latency is killing you.
- **Use torch.compile** for general PyTorch models. It is the superior choice for most users because it provides **kernel fusion**—which reduces actual memory movement—something CUDA Graphs cannot do on their own.

---

## Pro tip (NVIDIA)

**torch.compile** actually uses CUDA Graphs under the hood when you use `mode="reduce-overhead"`. It fuses the math first, then records the result as a graph.
