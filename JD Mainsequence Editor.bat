@echo off
cd /d "%~dp0"
start "" pythonw mainseq_editor.py %*
if errorlevel 1 python mainseq_editor.py %*
