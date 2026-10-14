@echo off
rem Downloads the Ternary-Bonsai-2-27B-PTQ1_0 GGUF (5.6 GB) into the repo root.
setlocal
set MODEL=https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf
set OUT=%~dp0..\Ternary-Bonsai-2-27B-PTQ1_0.gguf
echo Downloading Tertiary-Bonsai PTQ1_0 -> %OUT%
curl -L -o "%OUT%" "%MODEL%"
echo Done. Size:
dir "%OUT%"
endlocal
