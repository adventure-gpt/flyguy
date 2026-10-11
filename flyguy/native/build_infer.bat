@echo off
rem flyguy: the bot's compiled forward pass (CPU). Visual Studio Build Tools only.
rem   flyguy\native\build_infer.bat      -> flyguy\native\fly_infer.dll   (flyguy\fly.py uses it when it is there)
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64.bat not found & exit /b 1 )
cl /nologo /O2 /arch:AVX2 /fp:fast /EHsc /LD fly_infer.cpp /Fe:fly_infer_new.dll >nul
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
rem (a running bot keeps the old DLL open: the new one takes its place only if that works)
move /y fly_infer_new.dll fly_infer.dll >nul 2>&1
if errorlevel 1 ( echo built fly_infer_new.dll - fly_infer.dll is in use by a running bot, replace it when the bot has stopped & exit /b 0 )
del fly_infer_new.exp fly_infer_new.lib fly_infer.obj >nul 2>&1
echo built %cd%\fly_infer.dll
