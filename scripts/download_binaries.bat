@echo off
rem Downloads the PrismML llama.cpp fork CUDA-12.4 windows binaries + cudart
rem into a local "bin124" dir, and unpacks them next to this script.
setlocal
set REL=https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10777-e3de715
set ROOT=%~dp0..
set DEST=%ROOT%\bin124

echo [1/3] downloading binaries archive...
curl -L -o "%ROOT%\llama-cuda12.4.zip" "%REL%/llama-prism-b10777-e3de715-bin-win-cuda-12.4-x64.zip"
mkdir "%DEST%" 2>nul
tar -xf "%ROOT%\llama-cuda12.4.zip" -C "%DEST%"

echo [2/3] downloading CUDA 12.4 runtime DLLs...
curl -L -o "%ROOT%\cudart124.zip" "%REL%/cudart-llama-bin-win-cuda-12.4-x64.zip"
tar -xf "%ROOT%\cudart124.zip" -C "%DEST%"

echo [3/3] cleaning up archives...
del "%ROOT%\llama-cuda12.4.zip" "%ROOT%\cudart124.zip" 2>nul

echo Binaries ready in %DEST%. Verify:
dir "%DEST%\llama-server.exe"
endlocal
