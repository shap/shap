@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\Users\hp\gsoc\ESoC\shap\poc_nanobind
pip install . -v
echo.
echo === BUILD COMPLETE ===
pause
