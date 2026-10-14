#!/usr/bin/env python3
"""Quick smoke test against a running Bonsai server using its OpenAI endpoint."""
import json, sys, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"

def call(path, body=None):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)

models = call("/v1/models")
print("models:", [m["id"] for m in models["data"]], "n_ctx:", models["data"][0]["meta"]["n_ctx"])

r = call("/v1/chat/completions", {
    "model": "bonsai",
    "messages": [{"role": "user", "content": "What is 2+2? One word."}],
    "max_tokens": 64,
})
print("reply:", r["choices"][0]["message"]["content"])
print("tokens:", r["usage"])
