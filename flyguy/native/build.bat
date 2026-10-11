@echo off
rem flyguy native trainer: compile on the desktop (CUDA toolkit + Visual Studio Build Tools). CPU only.
rem   flyguy\native\build.bat      -> flyguy\native\flyguy_train.exe
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64.bat not found & exit /b 1 )
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -Xcompiler "/EHsc /W1" -o flyguy_train.exe flyguy_train.cu
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
echo built %cd%\flyguy_train.exe
