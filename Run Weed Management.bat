```bat
@echo off
title AI-Based Precision Weed Management

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" weed_ai

cd /d "%USERPROFILE%\weed_management"

streamlit run app.py

pause
```
