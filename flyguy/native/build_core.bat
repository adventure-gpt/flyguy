@echo off
rem flyguy: the native network as a library for the Python trainer (reinforcement learning; flyguy\fly_native.py).
rem Compile on the desktop (CUDA toolkit + Visual Studio Build Tools), CPU only; copy the DLL to the second PC.
rem   flyguy\native\build_core.bat      -> flyguy\native\flyguy_core.dll
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64.bat not found & exit /b 1 )
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -DFLYGUY_CORE -shared -Xcompiler "/EHsc /W1" -o flyguy_core_new.dll flyguy_train.cu
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
move /y flyguy_core_new.dll flyguy_core.dll >nul 2>&1
if errorlevel 1 ( echo built flyguy_core_new.dll - flyguy_core.dll is in use, replace it when the trainer has stopped & exit /b 0 )
del flyguy_core_new.exp flyguy_core_new.lib >nul 2>&1
echo built %cd%\flyguy_core.dll
