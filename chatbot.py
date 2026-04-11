import base64
import html
import json
import os
import re
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "shinzogpt-logo.svg"
MAX_PROMPT_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
MAX_UPLOAD_FILE_MB = int(os.getenv("MAX_UPLOAD_FILE_MB", "20"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))
HTTP_CONNECT_TIMEOUT_SEC = float(os.getenv("HTTP_CONNECT_TIMEOUT_SEC", "6"))
HTTP_READ_TIMEOUT_CHAT_SEC = float(os.getenv("HTTP_READ_TIMEOUT_CHAT_SEC", "180"))
HTTP_READ_TIMEOUT_UPLOAD_SEC = float(os.getenv("HTTP_READ_TIMEOUT_UPLOAD_SEC", "360"))

PROVIDER_OPTIONS = ["Groq", "Moonshot Kimi"]
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
MOONSHOT_MODELS = ["moonshot-v1-8k", "moonshot-v1-32k"]
MOONSHOT_NVIDIA_MODELS = ["moonshotai/kimi-k2-thinking"]
ROUTING_OPTIONS = ["Auto", "Chat only", "RAG only"]
ROUTING_MODE_MAP = {"Auto": "auto", "Chat only": "chat_only", "RAG only": "rag_only"}
TOOL_LABELS = {
    "calculator": "Calculator",
    "current_time": "Current Time",
    "weather": "Weather",
    "asset_price": "Price",
    "news": "News",
    "web_search": "Web Search",
}
AVAILABLE_TOOLS = [
    "Current Time",
    "Weather",
    "Price",
    "News",
    "Web Search",
    "Calculator",
]


def build_http_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP_SESSION = build_http_session()


def parse_backend_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message")
            if detail:
                return str(detail)
    except ValueError:
        pass
    return response.text or f"HTTP {response.status_code}"


def resolve_provider_config(provider: str):
    if provider == "Groq":
        return os.getenv("GROQ_API_KEY"), False, "Model", GROQ_MODELS

    moonshot_key = os.getenv("MOONSHOT_API_KEY")
    is_nvidia_key = bool(moonshot_key and moonshot_key.startswith("nvapi-"))
    if is_nvidia_key:
        return moonshot_key, True, "Model (NVIDIA)", MOONSHOT_NVIDIA_MODELS
    return moonshot_key, False, "Model", MOONSHOT_MODELS


def load_logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None
    try:
        svg_text = LOGO_PATH.read_text(encoding="utf-8")
        encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"
    except OSError:
        return None


def normalize_message_content(role: str, content: str) -> str:
    text = content or ""

    # Defensive cleanup: older custom markup could leak a trailing literal </div> in user messages.
    if role == "user":
        text = re.sub(r"(?:\r?\n)?\s*</div>\s*$", "", text, count=1, flags=re.IGNORECASE)

    return text


def fetch_runtime_summary():
    try:
        response = HTTP_SESSION.get(
            f"{API_URL}/metrics/summary",
            timeout=(HTTP_CONNECT_TIMEOUT_SEC, 10),
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def format_usd_value(value) -> str:
    if value in (None, "", "n/a"):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    formatted = f"{number:.8f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted = f"{formatted}.00"
    return formatted


def build_history_payload(chat_history: list[dict]) -> list[dict]:
    history_payload = []
    for message in chat_history[-MAX_HISTORY_TURNS:]:
        role = (message.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = normalize_message_content(role, message.get("content", "")).strip()
        if not content:
            continue
        history_payload.append({"role": role, "content": content[:MAX_PROMPT_CHARS]})
    return history_payload


def build_message_html(
    role: str,
    content: str,
    meta: str | None = None,
    is_latest: bool = False,
    extra_classes: str = "",
) -> str:
    role_class = "user" if role == "user" else "assistant"
    latest_class = " new" if is_latest else ""
    extra_class = f" {extra_classes.strip()}" if extra_classes.strip() else ""
    clean_content = normalize_message_content(role, content)
    safe_content = html.escape(clean_content).replace("\n", "<br>")
    meta_html = f'<div class="sg-meta">{html.escape(meta)}</div>' if meta else ""

    return (
        f'<div class="sg-row {role_class}">'
        f'<div class="sg-msg {role_class}{latest_class}{extra_class}">'
        '<div class="sg-body">'
        f'<div class="sg-text">{safe_content}</div>'
        f"{meta_html}"
        "</div>"
        "</div>"
        "</div>"
    )


def render_message(role: str, content: str, meta: str | None = None, is_latest: bool = False) -> None:
    st.markdown(build_message_html(role, content, meta=meta, is_latest=is_latest), unsafe_allow_html=True)


def render_message_into(
    container,
    role: str,
    content: str,
    meta: str | None = None,
    is_latest: bool = False,
    extra_classes: str = "",
) -> None:
    container.markdown(
        build_message_html(role, content, meta=meta, is_latest=is_latest, extra_classes=extra_classes),
        unsafe_allow_html=True,
    )


def render_typing_indicator() -> None:
    st.markdown(
        """
        <div class="sg-row assistant">
            <div class="sg-msg assistant typing new">
                <div class="sg-body">
                    <div class="typing-dots" aria-label="Assistant is thinking">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_typing_indicator_into(container) -> None:
    container.markdown(
        """
        <div class="sg-row assistant">
            <div class="sg-msg assistant typing new">
                <div class="sg-body">
                    <div class="typing-dots" aria-label="Assistant is thinking">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def iter_stream_events(response: requests.Response):
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def build_message_meta(
    provider: str, selected_model: str, backend_metrics: dict, route_used: str, latency_ms: int
) -> str:
    tool_used = str(backend_metrics.get("tool_used", "none")).strip().lower()
    if route_used == "rag":
        mode = "RAG"
    elif route_used == "chat_tools":
        mode = f"Chat + {TOOL_LABELS.get(tool_used, tool_used.replace('_', ' ').title())}"
    else:
        mode = "Chat"

    server_latency = backend_metrics.get("latency_ms", latency_ms)
    input_tokens = backend_metrics.get("estimated_input_tokens", "-")
    output_tokens = backend_metrics.get("estimated_output_tokens", "-")
    cost_usd = format_usd_value(backend_metrics.get("estimated_cost_usd"))
    cost_label = f"${cost_usd}" if cost_usd != "n/a" else "n/a"
    return (
        f"{provider} | {selected_model} | {mode} | {server_latency} ms"
        f" | in:{input_tokens} tok | out:{output_tokens} tok | {cost_label}"
    )


def render_chat_bottom_anchor() -> None:
    st.markdown('<div id="chat-bottom-anchor" class="chat-bottom-anchor"></div>', unsafe_allow_html=True)


def render_motion_bridge(auto_scroll: bool = False) -> None:
    if not auto_scroll:
        return

    components.html(
        """
        <script>
        const root = window.parent.document;
        const anchor = root.getElementById("chat-bottom-anchor");
        if (anchor) {
            const followForMs = 2200;
            const started = performance.now();
            let lastScroll = 0;

            const tick = (now) => {
                if (now - lastScroll > 140) {
                    anchor.scrollIntoView({
                        behavior: now - started < 260 ? "smooth" : "auto",
                        block: "end",
                    });
                    lastScroll = now;
                }
                if (now - started < followForMs) {
                    requestAnimationFrame(tick);
                }
            };

            requestAnimationFrame(tick);
        }
        </script>
        """,
        height=0,
        width=0,
    )


page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else ":robot_face:"
st.set_page_config(page_title="ShinzoGPT", page_icon=page_icon, layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg-base: #060606;
        --bg-panel: #121212;
        --bg-panel-soft: #171717;
        --bg-pill: #1c1c1c;
        --txt-main: #f1f1f1;
        --txt-muted: #a4a4a4;
        --line: rgba(255, 255, 255, 0.13);
        --line-strong: rgba(255, 255, 255, 0.24);
    }

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
        color: var(--txt-main);
        scroll-behavior: smooth;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1px 1px at 40px 90px, rgba(255, 255, 255, 0.40), transparent 100%),
            radial-gradient(1px 1px at 120px 40px, rgba(255, 255, 255, 0.28), transparent 100%),
            radial-gradient(1.2px 1.2px at 220px 160px, rgba(255, 255, 255, 0.36), transparent 100%),
            radial-gradient(1px 1px at 300px 100px, rgba(255, 255, 255, 0.30), transparent 100%),
            radial-gradient(1.3px 1.3px at 430px 230px, rgba(255, 255, 255, 0.35), transparent 100%),
            radial-gradient(1.1px 1.1px at 510px 60px, rgba(255, 255, 255, 0.28), transparent 100%),
            radial-gradient(800px 300px at 50% -180px, rgba(255, 255, 255, 0.09), transparent 70%),
            linear-gradient(180deg, #040404 0%, #0a0a0a 100%);
        background-size:
            260px 260px,
            330px 330px,
            420px 420px,
            500px 500px,
            640px 640px,
            760px 760px,
            100% 100%,
            100% 100%;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 8, 8, 0.62);
        backdrop-filter: blur(8px);
    }

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div {
        background: transparent !important;
        box-shadow: none !important;
        border-top: none !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 980px;
        padding-top: 2.3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(14, 14, 16, 0.98), rgba(10, 10, 12, 0.98));
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        margin-bottom: 1.05rem;
        background:
            linear-gradient(130deg, rgba(255, 255, 255, 0.05), transparent 40%),
            linear-gradient(180deg, rgba(18, 18, 18, 0.97), rgba(11, 11, 11, 0.97));
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.38);
        display: flex;
        align-items: center;
        gap: 0.95rem;
        animation: sg-fade-up 260ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
        transition: box-shadow 200ms ease, border-color 200ms ease;
    }

    .hero-center {
        max-width: 780px;
        margin: 0 auto 0.8rem;
    }

    .hero:hover {
        border-color: var(--line-strong);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.43);
    }

    .hero-logo {
        width: 62px;
        height: 62px;
        border-radius: 14px;
        border: 1px solid var(--line-strong);
        background: #0f0f0f;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: inset 0 0 22px rgba(255, 255, 255, 0.05);
    }

    .hero-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .hero-logo-fallback {
        font-family: "Space Grotesk", sans-serif;
        font-weight: 700;
        font-size: 1.08rem;
        letter-spacing: 0.5px;
        color: #d8d8d8;
    }

    .hero h1 {
        margin: 0;
        font-family: "Space Grotesk", sans-serif;
        font-size: 2rem;
        letter-spacing: 0.2px;
        color: var(--txt-main);
    }

    .hero p {
        margin: 0.3rem 0 0;
        color: var(--txt-muted);
        font-size: 0.94rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        background:
            linear-gradient(180deg, rgba(20, 20, 20, 0.94), rgba(12, 12, 12, 0.95));
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.25);
        transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--line-strong) !important;
        box-shadow: 0 14px 26px rgba(0, 0, 0, 0.32);
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-chat_prompt) {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-chat_prompt):hover {
        border: none !important;
        box-shadow: none !important;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin-bottom: 0.68rem;
    }

    .chip-row-center {
        justify-content: center;
        margin-bottom: 1rem;
    }

    .chip {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.27rem 0.66rem;
        font-size: 0.77rem;
        color: #dedede;
        background: var(--bg-pill);
        transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
    }

    .chip:hover {
        border-color: var(--line-strong);
        transform: translateY(-1px);
    }

    .chip.ok {
        border-color: rgba(255, 255, 255, 0.22);
        color: #f1f1f1;
    }

    .chip.warn {
        border-color: rgba(170, 170, 170, 0.34);
        color: #cfcfcf;
    }

    .empty-state {
        border: 1px dashed var(--line);
        border-radius: 14px;
        padding: 0.95rem 1.05rem;
        margin-top: 0.25rem;
        color: var(--txt-muted);
        background: rgba(255, 255, 255, 0.01);
    }

    .prompt-hint {
        color: var(--txt-muted);
        text-align: center;
        font-size: 0.88rem;
        margin: 0.2rem 0 0.55rem;
        animation: sg-fade-soft 320ms ease both;
    }

    .small-muted {
        color: var(--txt-muted);
        font-size: 0.84rem;
    }

    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 0.42rem 0.46rem 0.12rem;
        background:
            radial-gradient(circle at top, rgba(255, 255, 255, 0.06), transparent 62%),
            linear-gradient(180deg, rgba(24, 24, 28, 0.96), rgba(14, 14, 18, 0.97));
        box-shadow:
            0 10px 22px rgba(0, 0, 0, 0.24),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin: 0 auto 0.9rem;
        max-width: 900px;
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
        animation: sg-composer-in 260ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
    }

    [data-testid="stForm"]:focus-within {
        border-color: var(--line-strong);
        box-shadow:
            0 16px 30px rgba(0, 0, 0, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.06);
        transform: translateY(-1px);
    }

    [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
        align-items: center;
        gap: 0.5rem;
    }

    [data-testid="stForm"] [data-testid="column"] {
        display: flex;
        align-items: center;
    }

    [data-testid="stForm"] [data-testid="stTextInput"] {
        margin-bottom: 0 !important;
    }

    [data-testid="stForm"] .stFormSubmitButton {
        width: 100%;
        margin-top: 0 !important;
    }

    [data-testid="stTextInput"] > div > div > input {
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        background: rgba(23, 24, 29, 0.96) !important;
        min-height: 46px !important;
        padding-left: 1rem !important;
        color: #f0f2f6 !important;
        font-size: 1rem !important;
        transition: border-color 180ms ease, background 180ms ease, box-shadow 180ms ease !important;
    }

    [data-testid="stTextInput"] > div > div > input:focus {
        border-color: rgba(255, 255, 255, 0.26) !important;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08) !important;
    }

    .stFormSubmitButton > button {
        min-height: 46px;
        border-radius: 999px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.14);
        background:
            linear-gradient(180deg, rgba(36, 36, 38, 0.98), rgba(20, 20, 22, 0.98));
        color: #f2f2f2;
        box-shadow:
            0 8px 18px rgba(0, 0, 0, 0.22),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease, background 180ms ease;
    }

    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(255, 255, 255, 0.24);
        background:
            linear-gradient(180deg, rgba(42, 42, 44, 0.98), rgba(24, 24, 26, 0.98));
        box-shadow:
            0 12px 22px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.07);
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: #1d1d1d;
        color: #efefef;
        transition: border-color 180ms ease, background 180ms ease, transform 180ms ease, box-shadow 180ms ease;
    }

    .stButton > button:hover {
        border-color: var(--line-strong);
        background: #232323;
        transform: translateY(-1px);
    }

    .stButton > button:focus,
    .stButton > button:focus-visible {
        box-shadow: none !important;
        border-color: var(--line-strong);
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: #181818;
        transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: var(--line-strong);
        background: #1d1d1d;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed var(--line);
        border-radius: 12px;
        padding: 0.35rem 0.44rem 0.15rem;
        background: rgba(255, 255, 255, 0.012);
        transition: border-color 180ms ease, background 180ms ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--line-strong);
        background: rgba(255, 255, 255, 0.03);
    }

    [data-testid="stChatInput"] {
        position: relative;
        bottom: auto;
        padding: 0;
        margin: 0.45rem 0 0.3rem;
        z-index: 1;
        background: none !important;
    }

    [data-testid="stChatInput"] > div {
        width: min(100%, 880px);
        margin: 0 auto;
        min-height: 0;
        border: none !important;
        border-radius: 0;
        background: transparent !important;
        box-shadow: none !important;
        transition: none;
        padding: 0 !important;
        backdrop-filter: none;
    }

    [data-testid="stChatInput"] > div > div:last-child {
        display: flex;
        align-items: center;
        gap: 0.42rem;
        width: 100%;
        min-height: auto;
        border: none !important;
        border-radius: 0;
        background: transparent !important;
        box-shadow: none !important;
        transition: none;
        padding: 0;
        backdrop-filter: none;
    }

    [data-testid="stElementContainer"].st-key-chat_prompt {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        margin-top: 0.25rem;
    }

    [data-testid="stElementContainer"].st-key-chat_prompt > div {
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Streamlit changes the internal chat input DOM across releases, so avoid hiding
       positional children. Keep the actual composer visible and style stable. */
    [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-testid="stChatInputTextArea"] {
        display: flex !important;
    }

    [data-testid="stChatInput"] [data-baseweb="textarea"] {
        flex: 1 1 auto;
        min-width: 0;
        width: 100%;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 0 0 0.72rem;
    }

    [data-testid="stChatInput"] [data-baseweb="textarea"] > div {
        background: transparent !important;
    }

    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        flex: 1 1 auto;
        width: 100%;
        min-width: 0;
    }

    [data-testid="stChatInput"] [data-testid="stChatInputTextArea"] {
        flex: 1 1 auto;
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }

    [data-testid="stChatInput"] > div > div:last-child > div:nth-child(3) {
        flex: 1 1 auto;
        min-width: 0;
    }

    [data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
        border-color: rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        display: block !important;
        color: #f2f2f2 !important;
        background: transparent !important;
        border: none !important;
        font-size: 1rem !important;
        line-height: 1.45 !important;
    }

    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] input:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] textarea {
        width: 100% !important;
        padding: 0.72rem 0.08rem 0.68rem 0 !important;
        min-height: 24px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: rgba(230, 232, 238, 0.54) !important;
    }

    [data-testid="stChatInput"] button {
        flex: 0 0 auto;
        width: 40px !important;
        height: 40px !important;
        margin-bottom: 0;
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: transparent !important;
        box-shadow: none !important;
        transition: border-color 160ms ease, background 160ms ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        border-color: rgba(255, 255, 255, 0.18) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }

    .sg-row {
        display: flex;
        width: 100%;
        margin-bottom: 0.68rem;
        animation: sg-fade-soft 220ms ease both;
    }

    .sg-row.user {
        justify-content: flex-end;
    }

    .sg-row.assistant {
        justify-content: flex-start;
    }

    .sg-msg {
        display: block;
        max-width: min(82%, 760px);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.62rem 0.82rem;
        background: #151515;
        transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease, transform 180ms ease;
    }

    .sg-msg:hover {
        border-color: var(--line-strong);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22);
        transform: translateY(-1px);
    }

    .sg-msg.new {
        animation: sg-fade-up 320ms cubic-bezier(0.18, 0.88, 0.24, 1) both;
    }

    .sg-msg.user {
        border-color: rgba(255, 255, 255, 0.2);
        background: #1c1c1c;
    }

    .sg-msg.assistant {
        border-color: rgba(255, 255, 255, 0.14);
        background: #141414;
    }

    .sg-msg.typing {
        min-width: 78px;
        padding: 0.7rem 0.82rem;
        box-shadow:
            0 10px 22px rgba(0, 0, 0, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .sg-msg.streaming {
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow:
            0 12px 26px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .sg-body {
        min-width: 0;
        margin: 0;
        padding: 0;
    }

    .sg-text {
        color: var(--txt-main);
        font-size: 1.03rem;
        line-height: 1.52;
        word-break: break-word;
        margin: 0;
        padding: 0;
        text-indent: 0;
    }

    .sg-msg.streaming .sg-text::after {
        content: "";
        display: inline-block;
        width: 0.48rem;
        height: 1.02em;
        margin-left: 0.18rem;
        vertical-align: -0.12em;
        border-radius: 2px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(181, 189, 205, 0.7));
        animation: sg-caret 0.95s steps(1) infinite;
        opacity: 0.9;
    }

    .sg-meta {
        margin-top: 0.36rem;
        color: #9a9a9a;
        font-size: 0.77rem;
    }

    .typing-dots {
        display: inline-flex;
        align-items: center;
        gap: 0.34rem;
        min-height: 18px;
    }

    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: rgba(241, 241, 241, 0.78);
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.14);
        animation: sg-bounce 1s infinite ease-in-out;
    }

    .chat-bottom-anchor {
        width: 100%;
        height: 1px;
        margin-top: 0.2rem;
    }

    .typing-dots span:nth-child(2) {
        animation-delay: 0.14s;
    }

    .typing-dots span:nth-child(3) {
        animation-delay: 0.28s;
    }

    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.16);
        border-radius: 8px;
        border: 2px solid transparent;
        background-clip: content-box;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.28);
        background-clip: content-box;
    }

    @keyframes sg-fade-up {
        from {
            opacity: 0;
            transform: translateY(10px) scale(0.985);
            filter: blur(2px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0);
        }
    }

    @keyframes sg-composer-in {
        from {
            opacity: 0;
            transform: translateY(-12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes sg-bounce {
        0%, 80%, 100% {
            opacity: 0.4;
            transform: translateY(0);
        }
        40% {
            opacity: 1;
            transform: translateY(-3px);
        }
    }

    @keyframes sg-caret {
        0%, 49% {
            opacity: 0.92;
        }
        50%, 100% {
            opacity: 0;
        }
    }

    @keyframes sg-fade-soft {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
            scroll-behavior: auto !important;
        }
    }

    @media (max-width: 980px) {
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.2rem;
        }

        .hero h1 {
            font-size: 1.7rem;
        }

        .hero-logo {
            width: 54px;
            height: 54px;
        }

        .sg-msg {
            max-width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db_path" not in st.session_state:
    st.session_state.vector_db_path = None
if "uploaded_sources" not in st.session_state:
    st.session_state.uploaded_sources = []
if "runtime_summary" not in st.session_state:
    st.session_state.runtime_summary = None
if "runtime_summary_ts" not in st.session_state:
    st.session_state.runtime_summary_ts = 0.0
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "should_scroll_to_bottom" not in st.session_state:
    st.session_state.should_scroll_to_bottom = False

logo_data_uri = load_logo_data_uri()
logo_markup = (
    f'<img src="{logo_data_uri}" alt="ShinzoGPT logo" />'
    if logo_data_uri
    else '<div class="hero-logo-fallback">SG</div>'
)

with st.sidebar:
    st.markdown("### Session Controls")
    provider = st.selectbox("Provider", PROVIDER_OPTIONS)
    api_key, is_nvidia_key, model_label, model_options = resolve_provider_config(provider)
    selected_model = st.selectbox(model_label, model_options)
    routing_mode_label = st.selectbox(
        "Routing",
        ROUTING_OPTIONS,
        index=0,
        help="Auto chooses between chat and RAG. Chat only ignores docs. RAG only forces doc-grounded answers.",
    )
    routing_mode = ROUTING_MODE_MAP[routing_mode_label]
    enable_tools = st.toggle(
        "Enable Agent Tools",
        value=True,
        help="Used in chat mode for live lookups and calculations. RAG answers still come from your uploaded docs.",
    )

    key_chip = "Connected" if api_key else "Missing"
    key_class = "ok" if api_key else "warn"
    tools_chip = "On" if enable_tools else "Off"
    tools_class = "ok" if enable_tools else "warn"
    st.markdown(
        (
            '<div class="chip-row">'
            f'<span class="chip {key_class}">API Key: {key_chip}</span>'
            f'<span class="chip {tools_class}">Tools: {tools_chip}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption("Available tools: " + ", ".join(AVAILABLE_TOOLS))

    st.markdown("---")
    st.markdown("### Knowledge")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_col, reset_col = st.columns(2)
    with process_col:
        process_docs = st.button(
            "Process PDFs",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )
    with reset_col:
        reset_chat = st.button("Reset Chat", use_container_width=True)

    if process_docs and uploaded_files:
        if len(uploaded_files) > MAX_UPLOAD_FILES:
            st.error(f"Upload limit exceeded. Max {MAX_UPLOAD_FILES} files.")
            st.stop()

        for file in uploaded_files:
            size_mb = len(file.getvalue()) / (1024 * 1024)
            if size_mb > MAX_UPLOAD_FILE_MB:
                st.error(f"{file.name} exceeds {MAX_UPLOAD_FILE_MB}MB limit.")
                st.stop()

        with st.spinner("Indexing documents..."):
            try:
                files_data = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                response = HTTP_SESSION.post(
                    f"{API_URL}/upload",
                    files=files_data,
                    timeout=(HTTP_CONNECT_TIMEOUT_SEC, HTTP_READ_TIMEOUT_UPLOAD_SEC),
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.vector_db_path = result.get("vector_db_path")
                    st.session_state.uploaded_sources = [f.name for f in uploaded_files]
                    st.success("Knowledge base updated.")
                else:
                    st.error(f"Backend failed: {parse_backend_error(response)}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    if reset_chat:
        st.session_state.chat_history = []
        st.session_state.vector_db_path = None
        st.session_state.uploaded_sources = []
        st.session_state.pending_prompt = None
        st.session_state.should_scroll_to_bottom = False
        st.rerun()

    rag_active = bool(st.session_state.vector_db_path)
    rag_chip = "RAG Active" if rag_active else "RAG Inactive"
    rag_class = "ok" if rag_active else "warn"
    doc_count = len(st.session_state.uploaded_sources)
    st.markdown(
        f"""
        <div class="chip-row">
            <span class="chip {rag_class}">{rag_chip}</span>
            <span class="chip">Docs: {doc_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.uploaded_sources:
        st.markdown("<div class='small-muted'>Indexed files:</div>", unsafe_allow_html=True)
        for doc_name in st.session_state.uploaded_sources[:5]:
            st.markdown(f"- {doc_name}")

    st.markdown("---")
    summary_refresh = st.button("Refresh Runtime Metrics", use_container_width=True)
    if summary_refresh or (time.time() - st.session_state.runtime_summary_ts) > 30:
        st.session_state.runtime_summary = fetch_runtime_summary()
        st.session_state.runtime_summary_ts = time.time()

    with st.expander("Runtime Metrics", expanded=False):
        summary = st.session_state.runtime_summary
        if summary:
            left_m, right_m = st.columns(2)
            left_m.metric("Requests", summary.get("requests_total", 0))
            right_m.metric("Errors", summary.get("errors_total", 0))
            left_m.metric("Avg Latency (ms)", summary.get("avg_request_latency_ms", 0))
            right_m.metric("Est. Cost (USD)", format_usd_value(summary.get("estimated_cost_usd_total", 0)))
        else:
            st.caption("Metrics unavailable.")

if routing_mode == "chat_only":
    mode_text = "Chat Only Mode"
    mode_class = "warn"
elif routing_mode == "rag_only":
    mode_text = "RAG Only Mode" if st.session_state.vector_db_path else "RAG Only (No Knowledge Base)"
    mode_class = "ok" if st.session_state.vector_db_path else "warn"
else:
    mode_text = "Using Knowledge Base" if st.session_state.vector_db_path else "General Chat Mode"
    mode_class = "ok" if st.session_state.vector_db_path else "warn"

st.markdown(
    f"""
    <div class="hero hero-center">
        <div class="hero-logo">{logo_markup}</div>
        <div>
            <h1>ShinzoGPT</h1>
            <p>Ask anything. Attach knowledge only when you want grounded answers.</p>
        </div>
    </div>
    <div class="chip-row chip-row-center">
        <span class="chip">Provider: {provider}</span>
        <span class="chip">Model: {selected_model}</span>
        <span class="chip">Routing: {routing_mode_label}</span>
        <span class="chip {'ok' if enable_tools else 'warn'}">Tools: {'On' if enable_tools else 'Off'}</span>
        <span class="chip {mode_class}">{mode_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

streaming_placeholder = None
with st.container():
    history_len = len(st.session_state.chat_history)
    for idx, message in enumerate(st.session_state.chat_history):
        if message.get("role") == "user":
            message["content"] = normalize_message_content("user", message.get("content", ""))
        render_message(
            message["role"],
            message["content"],
            message.get("meta"),
            is_latest=(idx == history_len - 1),
        )
    if st.session_state.pending_prompt:
        streaming_placeholder = st.empty()
        render_typing_indicator_into(streaming_placeholder)
    render_chat_bottom_anchor()

user_prompt = st.chat_input(
    "Ask ShinzoGPT...",
    key="chat_prompt",
    max_chars=MAX_PROMPT_CHARS,
    disabled=bool(st.session_state.pending_prompt),
)

if user_prompt:
    cleaned_prompt = normalize_message_content("user", user_prompt).strip()
    if not cleaned_prompt:
        st.rerun()
    if len(cleaned_prompt) > MAX_PROMPT_CHARS:
        st.error(f"Prompt too long. Max {MAX_PROMPT_CHARS} characters.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": cleaned_prompt})
    st.session_state.pending_prompt = cleaned_prompt
    st.session_state.should_scroll_to_bottom = True
    st.rerun()

render_motion_bridge(auto_scroll=st.session_state.should_scroll_to_bottom)

if st.session_state.pending_prompt:
    pending_prompt = str(st.session_state.pending_prompt)

    if not api_key:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"API key for {provider} is missing. Add it in Space secrets."}
        )
        st.session_state.pending_prompt = None
        st.session_state.should_scroll_to_bottom = True
        st.rerun()

    try:
        start = time.perf_counter()
        history_payload = build_history_payload(st.session_state.chat_history[:-1])
        payload = {
            "query": pending_prompt,
            "provider": provider,
            "model": selected_model,
            "api_key": api_key,
            "vector_db_path": st.session_state.vector_db_path,
            "is_nvidia_key": is_nvidia_key,
            "routing_mode": routing_mode,
            "enable_tools": enable_tools,
            "chat_history": history_payload,
        }
        bot_reply = ""
        meta = None
        route_used = "chat"
        stream_error = None

        with HTTP_SESSION.post(
            f"{API_URL}/chat/stream",
            json=payload,
            timeout=(HTTP_CONNECT_TIMEOUT_SEC, HTTP_READ_TIMEOUT_CHAT_SEC),
            stream=True,
        ) as response:
            if response.status_code != 200:
                stream_error = f"Backend error: {parse_backend_error(response)}"
            else:
                if streaming_placeholder is None:
                    streaming_placeholder = st.empty()
                render_typing_indicator_into(streaming_placeholder)

                for event in iter_stream_events(response):
                    event_type = str(event.get("type", "")).strip().lower()
                    if event_type == "chunk":
                        delta = str(event.get("delta", ""))
                        if not delta:
                            continue
                        bot_reply += delta
                        render_message_into(
                            streaming_placeholder,
                            "assistant",
                            bot_reply,
                            is_latest=True,
                            extra_classes="streaming",
                        )
                    elif event_type == "done":
                        route_used = str(event.get("route_used", "chat")).strip().lower() or "chat"
                        backend_metrics = event.get("metrics", {})
                        meta = build_message_meta(
                            provider,
                            selected_model,
                            backend_metrics if isinstance(backend_metrics, dict) else {},
                            route_used,
                            int((time.perf_counter() - start) * 1000),
                        )
                    elif event_type == "error":
                        stream_error = str(event.get("message", "Streaming response failed."))
                        break

        if stream_error:
            bot_reply = f"{bot_reply}\n\n{stream_error}".strip() if bot_reply else stream_error
        elif not bot_reply.strip():
            bot_reply = "No response from agent."

        if meta is None and bot_reply.strip():
            meta = build_message_meta(
                provider,
                selected_model,
                {},
                route_used,
                int((time.perf_counter() - start) * 1000),
            )

        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply, "meta": meta})
    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"Connection error: {e}. Is FastAPI running?"}
        )
    finally:
        st.session_state.pending_prompt = None
        st.session_state.should_scroll_to_bottom = True

    st.rerun()
st.session_state.should_scroll_to_bottom = False
