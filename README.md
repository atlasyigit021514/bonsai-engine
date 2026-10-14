# Bonsai Engine

Run **PrismML Ternary-Bonsai-2-27B** — a 27B-class *ternary (2-bit)* hybrid-attention LLM — on a single consumer GPU with as little as **8 GB VRAM**, served as a **localhost OpenAI-compatible API**.

The engine is the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (the only runtime with the custom **`PTQ1_0` / `PQ2_0` ternary + Walsh–Hadamard activation** CUDA kernels and **hybrid full/linear-attention** support). Stock llama.cpp refuses / garbles these model files.

> Hardware-tuned and measured on an **RTX 3070 (8 GB, sm_86)** — Windows, CUDA 12.x driver.

---

## Features

- **Real 27B-class model in 2-bit (1.75 bpw)** — ~98% of FP16 intelligence at ~5.6 GB (see [ptq-vs-pqq](#why-ptq1_0-over-pq2_0))
- **Native OpenAI-compatible endpoint** — drop into LangChain / Open WebUI / any OpenAI client (`/v1/models`, `/v1/chat/completions`)
- **Hybrid-attention aware** — 75 % linear-attention layers use a tiny recurrent state; only 16/64 layers hold real KV, which is what makes long context memory-feasible
- **Mixed-memory modes** — choose *speed* (KV on GPU) or *long context* (KV on CPU RAM)
- **128k window** available on 8 GB via 4-bit KV offload to system RAM

---

## Requirements

- A CUDA GPU with **≥ 8 GB VRAM** (Ampere/sm_86+ preferred — tuned kernels). *Tested: RTX 3070.*
- 16 GB+ system RAM for the long-context (KV-in-RAM) mode
- Windows (binaries) or build the fork from source per its README
- ~6 GB free disk for the model

---

## Quickstart

```bat
:: 1. clone & cd
:: 2. download the model
scripts\download_model.bat

:: 3. download the engine binaries (optional - they come from the fork release)
scripts\download_binaries.bat

:: 4. start the OpenAI-compatible server (32k ctx, KV on GPU - fast)
start_server.bat
```

Then:

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

Any OpenAI SDK works once you point `base_url` at `http://127.0.0.1:8080/v1`.

---

## Engine modes

| Mode | Ctx | KV location | Decode | Use |
|------|-----|-------------|--------|-----|
| `speed` (default) | 32k | GPU (q4_0) | **~45–50 t/s** | everyday, fast |
| `long` | 128k | CPU RAM (q4_0) | ~2–9 t/s | full window, max context |
| `bench` | auto | — | measures | tuning |

```bash
python bonsai_engine.py speed "Explain quantum entanglement briefly."
python bonsai_engine.py long  "Summarize this 128k document..."
python bonsai_engine.py bench
```

---

## Measured performance (RTX 3070, `-ngl 99`, flash-attn on)

| Metric | Result | Note |
|--------|--------|------|
| **Prefill** (pp512) | **893 t/s** | batch-verified |
| **Decode** (short ctx, GPU KV) | **50.25 t/s** | 67 % of the 448 GB/s wall |
| Decode (64k ctx, GPU KV) | ~12 t/s | full-attn is O(context)/token |
| Decode (128k ctx, KV in RAM) | ~2–9 t/s | full-attn reads window over PCIe |
| VRAM @32k (model+KV) | ~6.5 GiB / 8 | fits; ~1.5 GiB headroom |

---

## Why PTQ1_0 over PQ2_0

Two packs ship for this model:

| Pack | bpw | Size | Fits 8 GB + KV? |
|------|-----|------|-----------------|
| **PTQ1_0** | 1.75 | 5.95 GB | ✅ (this engine) |
| PQ2_0 | 2.13 | 7.21 GB | ❌ (~0 GB left on 8 GB) |

On a memory-tight card, **PTQ1_0 is required** and is also the *faster-decode* pick.

---

## The two hard limits (physics, not tuning)

1. **Decode is memory-bandwidth-bound.** Every token re-reads the ~5.6 GB model. On an RTX 3070 (448 GB/s), 70 t/s would need ~93 % of peak bandwidth for weights alone — physically unreachable; the tuned ternary kernel sustains ~50 t/s (67 %).
2. **128k KV can't fit VRAM beside the model.** 16 full-attn layers ⇒ 8.4 GB (bf16) / 2.1 GB (q4_0) at 128k. So long context lives in system RAM (PCIe-bound), or you trade window for speed.

**To hit 70 t/s + full 128k simultaneously:** a GPU with ≥ ~620 GB/s effective bandwidth and ≥ 16 GB VRAM (e.g. RTX 4080/4090). Same code, no changes.

### Roadmap: the theoretical speedup (not yet wired)
Batch-verification (speculative decoding / MTP heads) reads the weights **once per K tokens**, dividing the per-token bandwidth cost by K — the genuine path past 50 t/s on this card. Requires a draft model or MTP heads + a batched-verify path targetable at the `PTQ1_0` ternary backend, which the fork does not yet expose.

---

## Layout

```
bonsai_engine.py      # CLI launcher (speed / long / bench)
start_server.bat      # OpenAI-compatible server (32k, KV on GPU)
scripts/
  download_model.bat      # pulls Ternary-Bonsai-2-27B-PTQ1_0.gguf (5.6 GB)
  download_binaries.bat   # pulls the fork CUDA 12.4 win-x64 archive + cudart
```

The model (5.6 GB) and fork binaries (≈1.3 GB) are **not committed** — they are fetched by the scripts (GitHub has file-size limits).

---

## Attribution & license

- Model: [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) — see its [LICENSE](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/LICENSE) and `KNOWN_ISSUES.md`.
- Runtime: [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), branch `prism` (MIT).
- See [`LICENSE`](./LICENSE) for the project scaffolding.
