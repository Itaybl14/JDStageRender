@echo off
cd /d "%~dp0"
start "" pythonw stage_render_gui.py
if errorlevel 1 python stage_render_gui.py
