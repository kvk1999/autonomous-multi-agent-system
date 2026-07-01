@echo off
setlocal
cd /d "D:\Guvi FS Programs\Autonomous-Multi-Agent-System"
"%~dp0.venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501
