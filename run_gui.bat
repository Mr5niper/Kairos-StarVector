:: run_gui.bat (Windows helper - run from project root)
@echo off
setlocal
set PYTHONPATH=%CD%
streamlit run .\gui\streamlit_app.py
endlocal