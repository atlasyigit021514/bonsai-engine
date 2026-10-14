@echo off
rem Start the Bonsai OpenAI-compatible server (128k ctx; model on GPU, KV in RAM)
"C:\Users\atlas\bonsai\bin124\llama-server.exe" ^
  -m "C:\Users\atlas\bonsai\Ternary-Bonsai-2-27B-PTQ1_0.gguf" ^
  -c 131072 -ngl 99 -fa on -nkvo -ctk q4_0 -ctv q4_0 ^
  --host 127.0.0.1 --port 8080 --alias bonsai -t 6 --threads-http 2
