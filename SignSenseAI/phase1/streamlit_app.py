"""Streamlit web client for the shared SignSense AI FastAPI backend."""

import hashlib
import mimetypes
import os
from datetime import datetime

import httpx
import streamlit as st

from localization import strings_for


API_BASE_URL = os.getenv("SIGN_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SUPPORTED_TRANSLATION_TARGETS = {"English": "eng_Latn", "ஆங்கிலம்": "eng_Latn"}

st.set_page_config(
    page_title="SignSense AI | Tamil Signboard Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --blue: #2563EB;
        --blue-dark: #0F172A;
        --blue-soft: #DBEAFE;
        --blue-pale: #EFF6FF;
        --border: #D7E6FA;
        --muted: #526783;
        --white: #FFFFFF;
    }
    .stApp {
        background: var(--white);
        color: var(--blue-dark);
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.96);
    }
    [data-testid="stSidebar"], #MainMenu, footer {
        display: none;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 0.35rem;
        padding-bottom: 1.4rem;
    }
    [data-testid="stDivider"] {
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #C8DCF7 !important;
        color: #0F172A !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #0F172A !important;
    }
    div[role="radiogroup"] {
        gap: 0.35rem;
    }
    div[role="radiogroup"] label {
        padding: 0.45rem 0.55rem;
        border-bottom: 2px solid transparent;
    }
    div[role="radiogroup"] label p {
        color: #29496F !important;
        font-weight: 600;
    }
    div[role="radiogroup"] label:has(input:checked) {
        border-bottom-color: #2563EB;
    }
    div[role="radiogroup"] label:has(input:checked) p {
        color: #2563EB !important;
    }
    div[role="radiogroup"] label > div:first-child {
        display: none;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--blue);
        font-size: 1.15rem;
        font-weight: 750;
        white-space: nowrap;
        padding-top: 0.45rem;
    }
    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 0.65rem;
        background: var(--blue);
        color: white;
        font-size: 1.2rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(1.2rem, 3vw, 2rem);
        margin: 0.4rem 0 1.15rem;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: linear-gradient(105deg, #F2F8FF 0%, #EAF4FF 100%);
    }
    .hero-kicker {
        color: #3566A4;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0.45rem 0;
        color: var(--blue-dark);
        font-size: clamp(1.8rem, 4vw, 2.55rem);
        line-height: 1.15;
    }
    .hero p {
        color: #3E5878;
        margin: 0.35rem 0;
    }
    .hero-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(210px, .8fr);
        gap: 1rem;
        align-items: center;
    }
    .hero-art {
        min-height: 10rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #C8DDF8;
        border-radius: 16px;
        background: rgba(255,255,255,.66);
    }
    .hero-sign {
        padding: 1rem 1.6rem;
        border: 4px solid white;
        outline: 2px solid var(--blue);
        border-radius: 10px;
        background: var(--blue);
        box-shadow: 0 8px 22px rgba(37, 99, 235, .20);
        color: white;
        font-size: clamp(1.5rem, 3vw, 2.25rem);
        font-weight: 700;
    }
    .feature-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    .feature-chip {
        display: inline-flex;
        gap: 0.42rem;
        align-items: center;
        padding: 0.48rem 0.68rem;
        border: 1px solid #D8E8FC;
        border-radius: 10px;
        background: rgba(255,255,255,.78);
        color: #23466F;
        font-size: 0.8rem;
        font-weight: 600;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border);
        border-radius: 13px;
        background: #FFFFFF;
    }
    div.stButton > button {
        min-height: 2.8rem;
        border: 0;
        border-radius: 9px;
        background: var(--blue);
        color: white;
        font-weight: 650;
    }
    div.stButton > button:hover {
        border: 0;
        background: #1D4ED8;
        color: white;
    }
    div[data-testid="stFileUploader"] section {
        border-color: #B8D3F6;
        border-radius: 10px;
        background: #F8FBFF;
    }
    div[data-testid="stFileUploader"] section button {
        border: 1px solid #2563EB !important;
        border-radius: 8px !important;
        background: #2563EB !important;
        color: #FFFFFF !important;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] p {
        color: #334155 !important;
        opacity: 1 !important;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #0F172A !important;
        opacity: 1 !important;
    }
    .result-label {
        color: #31527A;
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 0.28rem;
    }
    .result-value {
        color: var(--blue-dark);
        font-size: 1rem;
        overflow-wrap: anywhere;
    }
    .category-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        background: #DDF7F0;
        color: #087653;
        font-size: 0.84rem;
        font-weight: 700;
    }
    .confidence-box {
        border: 1px solid #F6D889;
        border-radius: 10px;
        background: #FFFAEB;
        padding: 0.75rem 0.85rem;
        color: #624A13;
    }
    .muted {
        color: #334155;
        font-size: 0.84rem;
        opacity: 1;
    }
    .footer {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        align-items: center;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        border-top: 1px solid #E3EDF9;
        color: var(--muted);
        font-size: 0.78rem;
    }
    @media (max-width: 760px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.45rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }
        .hero {
            padding: 1rem;
        }
        .hero-art {
            min-height: 7rem;
        }
        .hero-layout {
            grid-template-columns: 1fr;
        }
        .feature-chip {
            font-size: 0.74rem;
        }
        div[role="radiogroup"] {
            flex-wrap: wrap;
            gap: 0.2rem;
        }
        .footer {
            align-items: flex-start;
            flex-direction: column;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

interface_language = st.session_state.get("interface_language", "English")
header_brand, header_nav, header_language = st.columns([2.2, 5.5, 1.7], vertical_alignment="center")
with header_brand:
    st.markdown(
        '<div class="brand"><span class="brand-mark">📷</span>SignSense AI</div>',
        unsafe_allow_html=True,
    )
with header_language:
    interface_language = st.selectbox(
        "Interface language",
        options=["English", "தமிழ்"],
        index=0 if interface_language == "English" else 1,
        label_visibility="collapsed",
        key="interface_language",
    )

text = strings_for(interface_language)
with header_language:
    try:
        health_response = httpx.get(f"{API_BASE_URL}/health", timeout=2)
        health_response.raise_for_status()
        api_health = health_response.json()
        status_text = (
            f"● {text['api_connected']} · "
            f"{text['translation_ready'] if api_health.get('translation_configured') else text['translation_not_configured']}"
        )
        st.caption(status_text)
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        st.caption(f"● {text['backend_offline']}")
navigation = text["navigation"]
page_labels_by_id = {
    "home": navigation["home"],
    "translate": navigation["translate"],
    "history": navigation["history"],
    "about": navigation["about"],
}
navigation_key = f"navigation_{interface_language}"
if navigation_key not in st.session_state:
    current_page_id = st.session_state.get("current_page_id", "home")
    st.session_state[navigation_key] = page_labels_by_id.get(
        current_page_id,
        navigation["home"],
    )
with header_nav:
    selected_page_label = st.radio(
        "Navigation",
        options=list(page_labels_by_id.values()),
        horizontal=True,
        label_visibility="collapsed",
        key=navigation_key,
    )
page_ids = {
    navigation["home"]: "home",
    navigation["translate"]: "translate",
    navigation["history"]: "history",
    navigation["about"]: "about",
}
page = page_ids[selected_page_label]
st.session_state["current_page_id"] = page
st.divider()


def render_hero():
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-kicker">{text["kicker"]}</div>
          <div class="hero-layout">
            <div>
              <h1>{text["heading"]}</h1>
              <p><strong>{text["subtitle"]}</strong></p>
              <p>{text["hero_description"]}</p>
              <div class="feature-row">
                <span class="feature-chip">📷 {text["feature_ocr"]}</span>
                <span class="feature-chip">🔤 {text["feature_translation"]}</span>
                <span class="feature-chip">🧠 {text["feature_context"]}</span>
                <span class="feature-chip">🔊 {text["feature_voice"]}</span>
              </div>
            </div>
            <div class="hero-art"><div class="hero-sign">தமிழ்&nbsp; →</div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def display_result(result: dict):
    with st.container(border=True):
        st.markdown(f"### ◈ &nbsp; {text['result']}")
        if result.get("tamil_text"):
            values = [
                (text["original_text"], result.get("tamil_text")),
                (text["transliteration"], result.get("transliteration")),
                (text["translation"], result.get("english_translation")),
                (text["context"], result.get("context")),
            ]
            for row in (values[:2], values[2:]):
                left, right = st.columns(2, gap="small")
                for column, (label, value) in zip((left, right), row):
                    with column:
                        with st.container(border=True):
                            st.markdown(f"**{label}**")
                            st.write(value if value else "—")
            category = result.get("category") or "other_unknown"
            with st.container(border=True):
                st.markdown(
                    f'<span class="category-pill">▣ &nbsp; {text["category"]}: '
                    f'{text["categories"].get(category, text["categories"]["other_unknown"])}</span>',
                    unsafe_allow_html=True,
                )
            confidence = result.get("confidence")
            confidence_label = (
                f"{confidence:.1f}/100"
                if confidence is not None
                else text["confidence_unavailable"]
            )
            score_note = (
                text["confidence_note"]
                if confidence is not None
                else text["review"]
            )
            st.markdown(
                f'<div class="confidence-box">⚠ &nbsp; <strong>{text["confidence"]}: '
                f'{confidence_label}</strong><br><span class="muted">{score_note}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"{text['processing_time']}: {result.get('processing_time_ms') or '—'} ms"
            )
        else:
            st.info(text["no_text"])

        if result.get("needs_review"):
            st.warning(text["review"])
        if result.get("warnings"):
            with st.container(border=True):
                st.markdown(f"**{text['warnings']}**")
                for warning in result["warnings"]:
                    st.caption(f"• {warning}")
        with st.expander(text["models_used"]):
            st.caption(
                f"{text['ocr_model']}: "
                f"{result.get('ocr_model') or text['model_unavailable']}"
            )
            st.caption(
                f"{text['translation_model']}: "
                f"{result.get('translation_model') or text['model_unavailable']}"
            )
            st.caption(
                f"{text['context_model']}: "
                f"{result.get('context_model') or text['model_unavailable']}"
            )


def analyze_uploaded_image(
    image_bytes: bytes,
    file_name: str,
    mime_type: str,
    target_language: str,
    transliteration_enabled: bool,
):
    try:
        with st.spinner(text["loading"]):
            response = httpx.post(
                f"{API_BASE_URL}/api/v1/translate",
                params={
                    "translation_language": target_language,
                    "transliteration_enabled": transliteration_enabled,
                },
                files={"file": (file_name, image_bytes, mime_type)},
                timeout=120,
            )
            response.raise_for_status()
        result = response.json()
        st.session_state["last_sign_result"] = result
        st.session_state["last_image_digest"] = hashlib.sha256(image_bytes).hexdigest()
        st.session_state["last_image_bytes"] = image_bytes
        st.session_state["last_image_name"] = file_name
        st.success(text["success"])
    except httpx.ConnectError:
        st.error(text["api_unavailable"])
    except httpx.TimeoutException:
        st.error(text["timeout"])
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", text["service_error"])
        except (ValueError, AttributeError):
            detail = text["service_error"]
        st.error(f"{text['service_error']} {detail}")
    except httpx.RequestError:
        st.error(text["api_unavailable"])
    except ValueError:
        st.error(text["service_error"])


if page in {"home", "translate"}:
    render_hero()
    if page == "home":
        st.markdown(f"#### {text['steps_heading']}")
        step_cols = st.columns(3)
        for column, icon, label in zip(
            step_cols,
            ("①", "②", "③"),
            (text["step_upload"], text["step_ocr"], text["step_understand"]),
        ):
            with column:
                st.markdown(f"**{icon} &nbsp; {label}**", unsafe_allow_html=True)
    left_panel, right_panel = st.columns([1.05, 0.95], gap="medium")
    with left_panel:
        with st.container(border=True):
            st.markdown(f"### ⊕ &nbsp; {text['upload_heading']}")
            st.caption(text["upload_instructions"])
            upload_column, camera_column = st.columns(2)
            with upload_column:
                uploaded_image = st.file_uploader(
                    text["upload"],
                    type=["jpg", "jpeg", "png", "webp"],
                    help=text["upload_help"],
                    key="sign_file_upload",
                )
            with camera_column:
                if st.session_state.get("camera_open", False):
                    camera_image = st.camera_input(
                        text["camera_input"],
                        key="sign_camera_capture",
                    )
                    if camera_image is not None:
                        st.session_state["captured_camera_image_bytes"] = (
                            camera_image.getvalue()
                        )
                    if st.button(
                        text["close_camera"],
                        key="close_sign_camera",
                        use_container_width=True,
                    ):
                        st.session_state["camera_open"] = False
                        st.rerun()
                elif st.button(
                    text["camera"],
                    key="open_sign_camera",
                    use_container_width=True,
                ):
                    st.session_state["camera_open"] = True
                    st.rerun()

            if uploaded_image is not None:
                image_bytes = uploaded_image.getvalue()
                image_name = uploaded_image.name
                image_mime_type = uploaded_image.type or "application/octet-stream"
            elif st.session_state.get("captured_camera_image_bytes"):
                image_bytes = st.session_state["captured_camera_image_bytes"]
                image_name = "camera-capture.jpg"
                image_mime_type = "image/jpeg"
            else:
                image_bytes = None
                image_name = "signboard.png"
                image_mime_type = "application/octet-stream"

            if image_bytes is not None:
                st.image(
                    image_bytes,
                    caption=text["original_image"],
                    use_container_width=True,
                )
                current_image_digest = hashlib.sha256(image_bytes).hexdigest()
            else:
                current_image_digest = None

            st.markdown(f"#### ⚙ &nbsp; {text['settings']}")
            setting_cols = st.columns(3)
            with setting_cols[0]:
                st.selectbox(
                    text["translation_language"],
                    options=list(SUPPORTED_TRANSLATION_TARGETS),
                    index=0,
                    key=f"target_language_{interface_language}",
                )
            with setting_cols[1]:
                st.selectbox(
                    text["source_language"],
                    options=[text["source_tamil"]],
                    index=0,
                    key=f"source_language_{interface_language}",
                )
            with setting_cols[2]:
                transliteration_enabled = st.selectbox(
                    text["transliteration_setting"],
                    options=[text["enabled"], text["disabled"]],
                    index=0,
                    key=f"transliteration_{interface_language}",
                ) == text["enabled"]

            if st.button(
                text["translate_button"],
                type="primary",
                use_container_width=True,
                disabled=image_bytes is None,
            ):
                analyze_uploaded_image(
                    image_bytes,
                    file_name=image_name,
                    mime_type=image_mime_type
                    or mimetypes.guess_type(image_name)[0]
                    or "application/octet-stream",
                    target_language=SUPPORTED_TRANSLATION_TARGETS[
                        st.session_state[f"target_language_{interface_language}"]
                    ],
                    transliteration_enabled=transliteration_enabled,
                )
    with right_panel:
        result = st.session_state.get("last_sign_result")
        if current_image_digest != st.session_state.get("last_image_digest"):
            result = None
        if result:
            st.image(
                st.session_state.get("last_image_bytes"),
                caption=text["original_image"],
                use_container_width=True,
            )
            display_result(result)
            if st.button(text["translate_another"], use_container_width=True):
                st.session_state.pop("last_sign_result", None)
                st.session_state.pop("last_image_bytes", None)
                st.rerun()
        else:
            with st.container(border=True):
                st.markdown(f"### ◈ &nbsp; {text['result']}")
                st.info(text["result_empty"])
                st.markdown(
                    f"""
                    <div class="muted">
                    <p>① {text["original_text"]}</p>
                    <p>② {text["transliteration"]}</p>
                    <p>③ {text["translation"]}</p>
                    <p>④ {text["context"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

elif page == "history":
    render_hero()
    st.subheader(text["history_heading"])
    history_col, clear_col = st.columns([4, 1])
    with clear_col:
        clear_history = st.button(text["clear_history"], use_container_width=True)
    if clear_history:
        try:
            response = httpx.delete(f"{API_BASE_URL}/api/v1/history", timeout=10)
            response.raise_for_status()
            st.success(text["history_cleared"])
        except httpx.RequestError:
            st.error(text["api_unavailable"])
    try:
        response = httpx.get(f"{API_BASE_URL}/api/v1/history", timeout=10)
        response.raise_for_status()
        history_items = response.json().get("items", [])
    except (httpx.RequestError, httpx.HTTPStatusError):
        history_items = []
        st.error(text["api_unavailable"])
    if history_items:
        for item in history_items:
            created = item.get("created_at", "")
            try:
                timestamp = datetime.fromisoformat(created).astimezone().strftime("%Y-%m-%d %H:%M")
            except ValueError:
                timestamp = created
            with st.container(border=True):
                st.markdown(f"**{timestamp} · {item.get('file_name', '')}**")
                with st.expander(text["view_result"]):
                    display_result(item.get("result", {}))
    else:
        st.info(text["history_empty"])

else:
    render_hero()
    st.subheader(text["about_heading"])
    st.write(text["about_problem"])
    st.markdown(f"**{text['technology']}**")
    st.write(
        "Streamlit · FastAPI · Python · OpenCV · Pillow · Tesseract Tamil OCR · "
        "IndicTrans2 adapter · indic-transliteration · SQLite"
    )
    st.markdown(f"**{text['workflow']}**")
    st.write(text["workflow_description"])
    st.caption(text["model_note"])

st.markdown(
    f'<div class="footer"><span>💡 &nbsp; {text["footer"]}</span>'
    f'<span>🔎 &nbsp; SignSense AI</span></div>',
    unsafe_allow_html=True,
)
