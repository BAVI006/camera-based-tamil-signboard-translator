# SignSense AI backend services

The backend folder contains shared image-preprocessing, OCR, translation,
transliteration, context, pipeline, evaluation, and FastAPI modules. Translation
requires a model or a trusted hosted endpoint to be configured; it is never
replaced with made-up output.

## Start the shared API (Windows Command Prompt)

Install the Python dependencies from the repository root:

```cmd
cd "C:\Users\bavit\OneDrive\Documents\Desktop\Tamil -Ai"
.\.venv\Scripts\activate
python -m pip install -r SignSenseAI\backend\requirements.txt
```

In a first Command Prompt window, start FastAPI:

```cmd
cd "C:\Users\bavit\OneDrive\Documents\Desktop\Tamil -Ai\SignSenseAI"
set "TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe"
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Check the API at <http://127.0.0.1:8000/docs> or run:

```cmd
curl http://127.0.0.1:8000/health
```

In a second Command Prompt window, start the existing Streamlit web client:

```cmd
cd "C:\Users\bavit\OneDrive\Documents\Desktop\Tamil -Ai"
.\.venv\Scripts\activate
python -m streamlit run SignSenseAI\phase1\streamlit_app.py
```

The client sends the selected photo to the shared API. Set
`SIGN_API_BASE_URL` if the API is running elsewhere. The Streamlit language
selector changes interface labels only; OCR remains Tamil and the only enabled
translation target is English.

The API stores translation text and metadata in SQLite, but never stores
uploaded photos. By default, the history database is
`%LOCALAPPDATA%\SignSenseAI\history.sqlite3` on Windows so database writes do
not trigger Uvicorn's source-code reloader. Set `SIGN_HISTORY_DB` to an
alternative SQLite file path if needed. History is available at
`GET /api/v1/history` and can be cleared with `DELETE /api/v1/history`.

## Phase 2: image preprocessing

`services/preprocessing.py` validates JPG, PNG, and WEBP uploads, applies EXIF
orientation, limits image size, and creates an OCR-ready copy. The original
image is returned separately and is not modified. Contrast enhancement and
gentle denoising are enabled by default. Grayscale, Otsu thresholding,
sharpening, and four-corner perspective correction are optional.

Run the tests from the `SignSenseAI` directory in Windows Command Prompt:

```cmd
cd "C:\Users\bavit\OneDrive\Documents\Desktop\Tamil -Ai\SignSenseAI"
..\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
```

Example use from Python:

```python
from pathlib import Path

from backend.services.preprocessing import preprocess_image

image_bytes = Path("signboard.jpg").read_bytes()
result = preprocess_image(image_bytes)
result.original.save("original_oriented.png")
result.processed.save("ocr_ready.png")
```

Perspective points are optional pixel coordinates on the resized image, in
top-left, top-right, bottom-right, bottom-left order. To avoid changing text
shape accidentally, perspective correction is not guessed automatically.

The standalone OCR test below compares original and preprocessed images. No real
signboard dataset is currently present, so no OCR accuracy claim is made.

## Phase 3: pretrained Tamil OCR

`services/ocr_service.py` provides a replaceable OCR-engine interface and a
Tesseract implementation. It supports Tamil (`tam`) and mixed Tamil-English
(`tam+eng`), checks that requested language data is installed, and returns
Tesseract word boxes and its TSV confidence (0-100). Blank, low-detail, and
low-confidence results are flagged for review. The confidence is an OCR engine
score, not a measured probability of correctness. When a large, strongly
colored sign panel is detected, OCR also checks that cropped panel so nearby
scenery and symbols do not overwhelm the text. Weak first readings get a sparse
text-layout retry, and word-box coordinates remain relative to the full image.

On Windows, install Tesseract using the UB Mannheim Windows installer:

<https://github.com/UB-Mannheim/tesseract/wiki>

During installation, include Tamil language data. In Command Prompt, verify:

```cmd
tesseract --version
tesseract --list-langs
```

The language list must include `tam`; `eng` is also needed for `tam+eng`.
Configure a non-default executable location in Command Prompt with:

```cmd
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Run the standalone original-versus-preprocessed comparison from the
`SignSenseAI` directory:

```cmd
..\.venv\Scripts\python.exe backend\scripts\test_ocr.py "C:\path\to\signboard.jpg" --language tam
```

For signs that combine Tamil and English, use `--language tam+eng`. The image
path must be replaced with a real photograph. This prints text, available
Tesseract confidence, word-box count, and review warnings for both images.

Run the automated tests:

```cmd
..\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
```

## Phase 4: translation and transliteration

The Tamil-to-Latin transliteration uses the open-source
`indic-transliteration` package and its documented Sanscript API. The selected
ITRANS scheme is a scholarly transliteration, not a phonetic English spelling.
The service keeps it separate from machine translation.

The original AI4Bharat IndicTrans2 adapter remains available, but its English
model repository returned HTTP 401 (gated access) on this machine, and the
IndicTransToolkit maintainers state that their Cython package is not tested on
Windows. Do not select IndicTrans2 unless you have obtained Hugging Face access
and run it in the supported Linux/WSL environment.

For local CPU translation, the app also supports the public
`facebook/nllb-200-distilled-600M` model. It is a separate pretrained
translation model, not an IndicTrans2 model. The Hugging Face model card lists
CC-BY-NC-4.0; use it only for permitted non-commercial/research purposes and
review its terms before a public or commercial deployment. Its weights are
large (about 2.6 GB in this setup). The provided Windows setup uses Ubuntu WSL
because IndicTransToolkit cannot be built natively on this Windows machine.

Run these commands in the Ubuntu WSL terminal from the `SignSenseAI` folder:

```bash
sudo apt update
sudo apt install -y build-essential python3.12-venv tesseract-ocr tesseract-ocr-tam libgl1
python3 -m venv ~/.venvs/signsense-ai
~/.venvs/signsense-ai/bin/python -m pip install --upgrade pip
~/.venvs/signsense-ai/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.venvs/signsense-ai/bin/python -m pip install -r backend/requirements.txt -r backend/requirements-translation.txt opencv-python-headless==4.10.0.84 Pillow==10.4.0 pytesseract==0.3.10
export TESSERACT_CMD=/usr/bin/tesseract
export SIGN_TRANSLATION_BACKEND=nllb
export SIGN_TRANSLATION_MODEL=facebook/nllb-200-distilled-600M
~/.venvs/signsense-ai/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The first NLLB inference downloads the weights and can take several minutes;
later requests reuse the model loaded in the API process. Keep the WSL API
terminal open. In another PowerShell terminal, start Streamlit from the
workspace root:

```powershell
$env:SIGN_API_BASE_URL = "http://127.0.0.1:8000"
& .\.venv\Scripts\python.exe -m streamlit run SignSenseAI\phase1\streamlit_app.py --server.port 8502
```

The FastAPI `/health` response reports whether a translation backend is
configured. It does not claim that weights have loaded or that a translation
has been evaluated. NLLB output still needs checking against real signboard
examples.

Official implementation and setup references:

- <https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface>
- <https://github.com/VarunGumma/IndicTransToolkit>

Alternatively, set `SIGN_TRANSLATION_BACKEND=http`,
`SIGN_TRANSLATION_API_URL`, and optionally `SIGN_TRANSLATION_API_KEY` for a
trusted, compatible endpoint. Its expected request/response JSON is documented
in `services/translation_service.py`. Do not use a provider until you have
reviewed its privacy and data-retention terms.

If no model/API backend is configured, the API returns `english_translation: null`,
a review warning, and the other available evidence. It does not fabricate
translations.

For exact Tamil phrases listed in `data/verified_phrases.json`, the pipeline
uses the reviewed English meaning and transliteration instead of model output,
because a model can mistranslate short sign text. A reviewed phrase may also
match a complete OCR line when a sign includes a name or other text on separate
lines. Explicit OCR spelling aliases are allowed only when they are listed in
the data file, and the response flags those results for review. The starter
entries include `மருத்துவமனை`, `குப்பை போடாதே`, `கருத்தரித்தல் மையம்`
("Fertility Centre"), and `அமைதியாக இருங்கள்` ("Please remain quiet"), with
observed OCR variants for the fertility-centre phrase.
Unrecognized phrases still rely on the configured translation model and must
be checked; OCR and translation are not guaranteed correct for every photo.

The result's model/method details distinguish Tesseract OCR, the configured
translation model (or a reviewed phrase-dictionary match), and rule-based
context. There is no custom-trained model or general-purpose vision-language
model in this baseline.

## Phase 5: context and categories

The first context stage classifies by keyword evidence stored in
`data/category_terms.json` and can look up reviewed phrases in
`data/verified_phrases.json`. This small seed is not a general translation
database. Unknown or low-confidence input is marked for review. Explanations
do not invent addresses, distances, hours, routes, or safety instructions.

## Phase 6: pipeline and API schema

`services/sign_pipeline.py` runs preprocessing, OCR on original and processed
images, chooses the higher-confidence non-empty result when possible, and then
requests translation, transliteration, and contextual classification.
`backend/main.py` exposes:

- `GET /health`
- `POST /api/v1/translate` with multipart image field `file` and optional
  `translation_language=eng_Latn`

The JSON response includes Tamil text, transliteration, nullable translation,
category, context, real Tesseract confidence where available, review status,
processing time, warnings, and matched category evidence.

## Evaluation workflow

See `data/dataset/README.md` for rights-aware collection, annotation, split,
duplicate checking, and the CER/WER comparison command. No dataset results have
been generated or claimed.
