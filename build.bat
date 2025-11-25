@echo off
odin build . -target:js_wasm32 -out:public/main.wasm -o:speed
pause