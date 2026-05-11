@echo off

set PATH=%~dp0..\build\Release;%PATH%

qwen-tts.exe ^
    --model ..\models\qwen-talker-1.7b-base-Q8_0.gguf ^
    --codec ..\models\qwen-tokenizer-12hz-Q8_0.gguf ^
    --ref-audio freeman.wav ^
    --ref-text freeman.txt ^
    --lang English ^
    -o clone.wav < prompt.txt

pause
