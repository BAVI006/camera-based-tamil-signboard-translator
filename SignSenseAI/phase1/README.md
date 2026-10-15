# SignSense AI web client

This folder contains the Streamlit web client. Image processing, OCR,
translation, transliteration, context, and history are handled by the shared
FastAPI backend under `backend/`.

## Start the complete local web workflow (Windows Command Prompt)

Install requirements once from the repository folder:

```cmd
cd "C:\Users\bavit\OneDrive\Documents\Desktop\Tamil -Ai"
.\.venv\Scripts\activate
python -m pip install -r SignSenseAI\phase1\requirements.txt
python -m pip install -r SignSenseAI\backend\requirements.txt
```

In the first Command Prompt, start the backend:

```cmd
cd "C:\Users\bavit\OneDrive\Documents\Desktop\Tamil -Ai\SignSenseAI"
set "TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe"
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second Command Prompt, start the website:

```cmd
cd "C:\Users\bavit\OneDrive\Documents\Desktop\Tamil -Ai"
.\.venv\Scripts\activate
python -m streamlit run SignSenseAI\phase1\streamlit_app.py
```

Open the local Streamlit URL shown in the terminal. Upload a JPG, PNG, or WEBP
photo, or capture one with the camera. Select **Read Tamil Text**. The UI
language changes labels only; Tamil remains the OCR source language. English
is currently the only enabled translation target.

Set `SIGN_API_BASE_URL` if FastAPI is running somewhere other than
`http://127.0.0.1:8000`. The app has no hardcoded translation example: without
a configured translation model or API, the translation card stays empty and a
warning is shown.
