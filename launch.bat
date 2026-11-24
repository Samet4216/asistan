@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
echo ===================================================
echo  Kisisel Uzman Yapay Zeka Asistani Baslatiliyor...
echo ===================================================
.venv\Scripts\python.exe -m personal_ai.app.cli
pause
