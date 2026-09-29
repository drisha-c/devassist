@echo off
cd /d C:\Users\NISHA\Projects\devassist
call venv\Scripts\activate.bat
uvicorn main:app --reload