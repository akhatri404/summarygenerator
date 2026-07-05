"""
TextLens — Professional Text Summarizer
========================================
Developed by Er Ashish KC Khatri | www.ashishkhatri.com.np

Features
--------
- Multiple summarization models (T5, DistilBART, BART-large-CNN)
- Input sources: paste text, upload file (.txt / .pdf / .docx), or fetch a URL
- Sentence-aware chunking so long documents are fully summarized
  (no silent truncation at the model's token limit)
- Length presets + fine-grained custom control
- Beam search / sampling decoding options
- Live document stats: words, sentences, reading time, compression ratio
- Summary history within the session
- One-click download of the summary (.txt / .md)

Install
-------
pip install streamlit transformers torch sentencepiece
# optional, for extra input sources:
pip install pypdf python-docx requests beautifulsoup4
"""

import math
import re
import time
from datetime import datetime
from io import BytesIO

import streamlit as st

# ---------------------------------------------------------------------------
# Optional dependencies — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import docx  # python-docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB = True
except ImportError:
    HAS_WEB = False


# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TextLens · Summarizer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .stat-card {
          background: var(--secondary-background-color, #f0f2f6);
          border-radius: 10px;
          padding: 0.9rem 1.1rem;
          text-align: center;
      }
      .stat-card .value { font-size: 1.5rem; font-weight: 700; }
      .stat-card .label { font-size: 0.78rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.06em; }
      .summary-box {
          border-left: 4px solid #4f8bf9;
          background: rgba(79, 139, 249, 0.06);
          border-radius: 6px;
          padding: 1.1rem 1.3rem;
          line-height: 1.7;
          font-size: 1.02rem;
      }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "T5 Small — fast, lightweight": {
        "id": "t5-small",
        "max_input_tokens": 512,
        "prefix": "summarize: ",
    },
    "DistilBART CNN — balanced (recommended)": {
        "id": "sshleifer/distilbart-cnn-12-6",
        "max_input_tokens": 1024,
        "prefix": "",
    },
    "BART Large CNN — best quality, slower": {
        "id": "facebook/bart-large-cnn",
        "max_input_tokens": 1024,
        "prefix": "",
    },
    "T5 Base — better than small, moderate speed": {
        "id": "t5-base",
        "max_input_tokens": 512,
        "prefix": "summarize: ",
    },
}

LENGTH_PRESETS = {
    "Brief (~2-3 sentences)": (30, 80),
    "Standard (~1 paragraph)": (80, 160),
    "Detailed (~2 paragraphs)": (160, 300),
    "Custom": None,
}


# ---------------------------------------------------------------------------
# Cached model loader  (st.cache_resource — NOT cache_data — for models)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_summarizer(model_id: str):
    from transformers import pipeline
    return pipeline("summarization", model=model_id, tokenizer=model_id)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


_ABBREVS = ("Dr", "Mr", "Mrs", "Ms", "St", "Jr", "Sr", "Prof",
            "vs", "etc", "Fig", "No", "Inc", "Ltd", "Co", "e.g", "i.e")
_PROTECT = "\u2402"  # rare placeholder char, restored after splitting

def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (avoids the nltk download step),
    tolerant of common abbreviations like Dr., Mr., etc."""
    protected = text
    for ab in _ABBREVS:
        protected = re.sub(rf"\b{re.escape(ab)}\.", ab + _PROTECT, protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", protected)
    return [p.replace(_PROTECT, ".").strip() for p in parts if p.strip()]


def chunk_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split text into chunks that fit the model's context, on sentence
    boundaries, so nothing is silently truncated."""
    budget = max_tokens - 32  # headroom for special tokens / task prefix
    sentences = split_sentences(text)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        n = len(tokenizer.encode(sent, add_special_tokens=False))
        if n > budget:  # pathological single "sentence" — hard-split it
            ids = tokenizer.encode(sent, add_special_tokens=False)
            for i in range(0, len(ids), budget):
                chunks.append(tokenizer.decode(ids[i : i + budget], skip_special_tokens=True))
            continue
        if current_len + n > budget and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += n

    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def text_stats(text: str) -> dict:
    words = len(text.split())
    sentences = len(split_sentences(text))
    return {
        "words": words,
        "chars": len(text),
        "sentences": sentences,
        "read_min": max(1, math.ceil(words / 200)),
    }


# ---------------------------------------------------------------------------
# Input extractors
# ---------------------------------------------------------------------------
def extract_from_upload(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()

    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        if not HAS_PDF:
            st.error("PDF support requires `pip install pypdf`.")
            return ""
        reader = PdfReader(BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        if not HAS_DOCX:
            st.error("Word support requires `pip install python-docx`.")
            return ""
        d = docx.Document(BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    st.error("Unsupported file type. Use .txt, .md, .pdf, or .docx.")
    return ""


def extract_from_url(url: str) -> str:
    if not HAS_WEB:
        st.error("URL fetching requires `pip install requests beautifulsoup4`.")
        return ""
    try:
        resp = requests.get(
            url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (TextLens Summarizer)"}
        )
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Could not fetch the URL: {e}")
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    container = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p.split()) > 4)
    return text or container.get_text(" ", strip=True)


# ---------------------------------------------------------------------------
# Summarization engine
# ---------------------------------------------------------------------------
def summarize(text: str, model_key: str, min_len: int, max_len: int,
              decoding: str, progress_cb=None) -> str:
    cfg = MODELS[model_key]
    pipe = load_summarizer(cfg["id"])
    tokenizer = pipe.tokenizer

    chunks = chunk_text(text, tokenizer, cfg["max_input_tokens"])
    n = len(chunks)

    # Distribute the requested length across chunks
    per_chunk_max = max(30, max_len // n) if n > 1 else max_len
    per_chunk_min = max(10, min(min_len // n if n > 1 else min_len, per_chunk_max - 10))

    gen_kwargs = {"do_sample": False, "num_beams": 4} if decoding.startswith("Beam") \
        else {"do_sample": True, "top_p": 0.92, "temperature": 0.8}

    partials = []
    for i, chunk in enumerate(chunks):
        # keep generation shorter than the input to avoid warnings on tiny chunks
        input_len = len(tokenizer.encode(chunk, add_special_tokens=False))
        c_max = min(per_chunk_max, max(20, int(input_len * 0.9)))
        c_min = min(per_chunk_min, max(5, c_max - 10))
        out = pipe(cfg["prefix"] + chunk, max_length=c_max, min_length=c_min,
                   truncation=True, **gen_kwargs)
        partials.append(out[0]["summary_text"].strip())
        if progress_cb:
            progress_cb((i + 1) / n)

    combined = " ".join(partials)

    # If we had many chunks, do one refinement pass over the stitched summary
    if n > 1:
        combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
        if combined_tokens > max_len * 1.3 and combined_tokens < cfg["max_input_tokens"]:
            out = pipe(cfg["prefix"] + combined, max_length=max_len,
                       min_length=min(min_len, max_len - 10),
                       truncation=True, **gen_kwargs)
            combined = out[0]["summary_text"].strip()

    return combined


# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    model_key = st.selectbox("Model", list(MODELS.keys()), index=0,
                             help="Larger models produce better summaries but load and run slower.")

    preset = st.radio("Summary length", list(LENGTH_PRESETS.keys()), index=1)
    if preset == "Custom":
        min_len, max_len = st.slider("Token range (min – max)", 10, 500, (60, 180), step=10)
    else:
        min_len, max_len = LENGTH_PRESETS[preset]
        st.caption(f"≈ {min_len}–{max_len} tokens")

    decoding = st.radio("Decoding strategy", ["Beam search (precise)", "Sampling (creative)"],
                        help="Beam search is deterministic and factual; sampling gives varied phrasing.")

    st.divider()
    st.markdown(
        "**TextLens** · developed by\n\n"
        "**Er Ashish KC Khatri**\n\n"
        "🌐 [ashishkhatri.com.np](https://www.ashishkhatri.com.np)\n\n"
        "📞 +977-9846262393"
    )


# ---------------------------------------------------------------------------
# UI — Main
# ---------------------------------------------------------------------------
st.title("🔎 TextLens")
st.caption("Professional text summarization — paste text, upload a document, or point at a URL.")

tab_paste, tab_file, tab_url = st.tabs(["✍️ Paste text", "📄 Upload file", "🔗 From URL"])

source_text = ""

with tab_paste:
    source_text_paste = st.text_area("Text to summarize", height=260,
                                     placeholder="Paste an article, report, or any long text here…")
    if source_text_paste:
        source_text = source_text_paste

with tab_file:
    uploaded = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx"])
    if uploaded is not None:
        extracted = clean_text(extract_from_upload(uploaded))
        if extracted:
            st.success(f"Extracted {len(extracted.split()):,} words from **{uploaded.name}**")
            with st.expander("Preview extracted text"):
                st.write(extracted[:2000] + ("…" if len(extracted) > 2000 else ""))
            source_text = extracted

with tab_url:
    url = st.text_input("Article URL", placeholder="https://example.com/article")
    if url and st.button("Fetch article", key="fetch"):
        st.session_state["url_text"] = clean_text(extract_from_url(url))
    if st.session_state.get("url_text"):
        st.success(f"Fetched {len(st.session_state['url_text'].split()):,} words")
        with st.expander("Preview fetched text"):
            st.write(st.session_state["url_text"][:2000] + "…")
        source_text = st.session_state["url_text"]

source_text = clean_text(source_text)

# Live input stats
if source_text:
    s = text_stats(source_text)
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, value) in zip(
        (c1, c2, c3, c4),
        [("Words", f"{s['words']:,}"), ("Characters", f"{s['chars']:,}"),
         ("Sentences", f"{s['sentences']:,}"), ("Reading time", f"{s['read_min']} min")],
    ):
        col.markdown(
            f'<div class="stat-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
if st.button("✨ Generate summary", type="primary", use_container_width=True):
    if not source_text:
        st.error("Please provide some text first — paste it, upload a file, or fetch a URL.")
    elif len(source_text.split()) < 30:
        st.warning("The text is very short (under 30 words). A summary won't add much value.")
    else:
        progress = st.progress(0.0, text="Loading model…")
        try:
            t0 = time.time()
            load_summarizer(MODELS[model_key]["id"])  # warm the cache with a visible message
            progress.progress(0.05, text="Summarizing…")

            summary = summarize(
                source_text, model_key, min_len, max_len, decoding,
                progress_cb=lambda f: progress.progress(0.05 + 0.95 * f, text="Summarizing…"),
            )
            elapsed = time.time() - t0
            progress.empty()

            st.session_state.setdefault("history", []).insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "model": model_key.split(" — ")[0],
                "summary": summary,
                "input_words": len(source_text.split()),
            })

            # ---- Results ----
            st.subheader("Summary")
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
            st.write("")

            in_words, out_words = len(source_text.split()), len(summary.split())
            r1, r2, r3 = st.columns(3)
            r1.metric("Summary length", f"{out_words} words")
            r2.metric("Compression", f"{(1 - out_words / in_words) * 100:.0f}%")
            r3.metric("Time taken", f"{elapsed:.1f}s")

            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download .txt", summary,
                               file_name="summary.txt", use_container_width=True)
            d2.download_button("⬇️ Download .md",
                               f"# Summary\n\n{summary}\n\n---\n*Generated with TextLens*",
                               file_name="summary.md", use_container_width=True)

        except Exception as e:
            progress.empty()
            st.error(f"Something went wrong while summarizing: {e}")

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.get("history"):
    st.divider()
    with st.expander(f"🕘 Session history ({len(st.session_state['history'])})"):
        for item in st.session_state["history"]:
            st.markdown(
                f"**{item['time']}** · {item['model']} · {item['input_words']:,} words in"
            )
            st.write(item["summary"])
            st.markdown("---")
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()
