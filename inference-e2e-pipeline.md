# Inference End-to-End Pipeline

End-to-end flow for LLM inference from prompt to output.

<img src="resources/inference-e2e-pipeline.png" alt="LLM inference pipeline: Prompt → Tokenization → GPU (Prefill, decode/generation, model weights) → De-tokenization → Output" width="800" />

---

## LLM Inference: Prefill vs. Decode

Two distinct phases of LLM inference: **Prefill** (input processing, KV cache creation) and **Decode** (token generation). Each phase has a different hardware bottleneck and drives different performance metrics.

```mermaid
graph TB
    subgraph Phase1["PHASE 1: PREFILL - Input Processing"]
        direction TB
        P_Input["<b>User Prompt</b><br/>Multiple Tokens Input"]
        P_Compute["<b>COMPUTE-BOUND</b><br/>Bottleneck: GPU Tensor Cores<br/><i>Math: Parallel GEMMs</i>"]
        KV_Init["<b>KV Cache Creation</b><br/>Store Keys/Values for Prompt"]
        First_Tok_Internal["<b>First Token Logic</b><br/>Calculated Internally"]

        P_Input --> P_Compute
        P_Compute --> KV_Init
        KV_Init --> First_Tok_Internal
    end

    subgraph Phase2["PHASE 2: DECODE - Token Generation"]
        direction TB
        D_Input["<b>Input Processing</b><br/>First/Last Token Received"]
        D_Memory["<b>MEMORY-BANDWIDTH BOUND</b><br/>Bottleneck: VRAM Speed<br/><i>Action: Load Weights + KV Cache</i>"]
        D_Output["<b>Visible Token Output</b><br/>Streams to User"]
        KV_Update["<b>KV Cache Update</b><br/>Add New Token K/V"]

        D_Input --> D_Memory
        D_Memory --> D_Output
        D_Output --> KV_Update
        KV_Update -.->|"Autoregressive Loop"| D_Input
    end

    First_Tok_Internal -->|"Internal Handoff"| D_Input

    subgraph Metrics["Key Performance Metrics"]
        direction LR
        M1["<b>TTFT</b><br/>Time to First Token"]
        M2["<b>TPOT / TPS</b><br/>Time Per Output Token"]
    end

    Phase1 -.->|Determines| M1
    Phase2 -.->|Determines| M2

    style Phase1 fill:#e1f5fe,stroke:#01579b
    style Phase2 fill:#fff3e0,stroke:#e65100
    style P_Compute fill:#b3e5fc,stroke:#0288d1,stroke-width:2px
    style D_Memory fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style KV_Init fill:#c8e6c9,stroke:#2e7d32
    style KV_Update fill:#c8e6c9,stroke:#2e7d32
    style First_Tok_Internal fill:#eee,stroke:#999
```

---

## Breakdown of the Phases

| Feature | Phase 1: Prefill | Phase 2: Decode |
|--------|-------------------|------------------|
| **Input** | The entire user prompt (many tokens). | A single token (received from internal handoff or previous loop). |
| **Processing** | All prompt tokens are processed simultaneously. | Tokens are generated one by one (autoregressive). |
| **Primary goal** | Understand context, build KV cache, and trigger generation. | Predict tokens and output visible text to the user. |
| **Bottleneck** | **Compute-bound:** GPU is busy doing heavy math (matrix multiplications). | **Memory-bound:** GPU spends more time moving data from VRAM than doing math. |
| **Key metric** | **TTFT** (Time to First Token): time until the first output of the decode phase appears. | **TPS** (Tokens Per Second): how fast the text “streams” to the screen. |

---

## Understanding the Latency Split

<img src="resources/latency-ttft-tpot.png" alt="Latency split: TTFT covers the prefill phase, TPOT measures time per output token during decoding" width="700" />

- **Time to First Token (TTFT):** The prefill phase runs entirely “behind the scenes.” The user only sees output once prefill completes and hands off to the first step of the decode phase. **Longer prompts increase this “invisible” processing time.** TTFT is the main latency users feel before any text appears.

- **Time Per Output Token (TPOT):** This is the visible “typing” phase. Once decode begins, it emits tokens one by one. Each step must load model weights and the KV cache, so this phase is limited by **how fast the GPU can move data (bandwidth)**, not by how fast it can do math.

---

## Parallelism Strategies per Phase

Different [parallelism strategies](parallelism.md) benefit each phase differently:

<img src="resources/parallelism-vs-phases.png" alt="Parallelism strategies vs inference phases: PP helps decode, TP and DP help prefill" width="500" />

| | [PP](parallelism.md#pipeline-parallelism-pp) | [TP](parallelism.md#tensor-parallelism-tp) | [DP](parallelism.md#fsdp-fully-sharded-data-parallel) |
|---|:---:|:---:|:---:|
| **Prefill** (compute-bound) | Neutral | Great | Great |
| **Decode** (memory-bound) | Great | Neutral | Neutral |

- **Prefill is compute-bound** — it processes the entire prompt in parallel. **TP** splits the matrix math across GPUs so each does less work, directly reducing TTFT. **DP** runs independent requests on separate replicas, boosting throughput. PP adds pipeline latency without helping the heavy compute.
- **Decode is memory-bandwidth-bound** — each token step loads weights and KV cache. **PP** shards the model by layers so each GPU holds fewer weights in memory, reducing the data movement bottleneck. TP and DP don't directly ease the per-token memory bandwidth pressure.

> See [Parallelism Strategies for Large Models](parallelism.md) for full definitions of PP, TP, and DP/FSDP.

---

## Why is the KV Cache so important?

Without the KV cache, the model would have to **re-process the entire prompt** (redo prefill) for every new token it generates. By saving the intermediate states (Keys and Values) during prefill, the model only needs to compute the math for the **newest token** and look up the rest in the cache. That makes decode much cheaper and keeps token-by-token generation feasible.
