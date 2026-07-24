@echo off
setlocal
set PYTHON_EXE=E:\Anaconda\envs\patchtst\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python
"%PYTHON_EXE%" "%~dp0server.py" %*
