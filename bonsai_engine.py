#!/usr/bin/env python3
"""Bonsai engine launcher — Ternary-Bonsai-2-27B via the PrismML llama.cpp fork.

The only working runtime for PTQ1_0/PQ2_0 ternary (Walsh-Hadamard + hybrid-attn)
kernels is the PrismML llama.cpp fork. This wrapper shells its llama-cli/bench.

Measured on this box (RTX 3070, driver 560.94 -> CUDA 12.6 max, uses cuda-12.4 pack):
  prefill (pp512, ngl99, fa):   ~893 tok/s
  decode  (tg128,  short ctx):  ~50 tok/s   (67% of the 448 GB/s BW wall; 70 = 93% -> unreachable)
  decode  (64k ctx, q4_0 KV):   ~12 tok/s   (full-attn layers are O(context) per token)
  128k ctx (q4_0 KV in RAM):    works, ~2 tok/s (PCIe KV reads)  -> -nkvo -c 131072

Modalities:
  speed : max decode+prefill, short VRAM-resident context
  long  : 128k context window (4-bit KV offloaded to system RAM)
"""
import argparse, subprocess, os, sys

BIN = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BIN, "Ternary-Bonsai-2-27B-PTQ1_0.gguf")
CLI  = os.path.join(BIN, "bin124", "llama-cli.exe")
BENCH= os.path.join(BIN, "bin124", "llama-bench.exe")

COMMON = ["-m", MODEL, "-ngl", "99", "-fa", "on"]

def mode_speed():
    # max tokens/s at VRAM-resident context
    return COMMON + ["-c", "32768", "-ctk", "q4_0", "-ctv", "q4_0", "-st"]

def mode_long():
    # full 128k window; KV lives in system RAM (slow decode but real 128k)
    return COMMON + ["-c", "131072", "-nkvo", "-ctk", "q4_0", "-ctv", "q4_0", "-st"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="speed", choices=["speed", "long", "bench"])
    ap.add_argument("prompt", nargs="*", default=["Explain quantum entanglement briefly."])
    ap.add_argument("-n", type=int, default=64)
    args = ap.parse_args()
    if args.mode == "bench":
        subprocess.run([BENCH, "-m", MODEL, "-ngl", "99", "-fa", "on", "-p", "512", "-n", "128"])
        return
    base = mode_long() if args.mode == "long" else mode_speed()
    cmd = [CLI, *base, "-n", str(args.n), "-p", " ".join(args.prompt)]
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
