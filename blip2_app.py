"""
BLIP-2 Interactive Walkthrough
SEAS 8525: Computer Vision and Generative AI
Dr. Elbasheer, Week 6

Run locally:
    pip install -r requirements.txt
    streamlit run blip2_app.py

Notes:
    Model: Salesforce/blip2-opt-2.7b (about 3.8B parameters).
    It loads only when a student presses a ▶ button; every diagram works without it.
    GPU recommended. On CPU it needs about 16 GB of RAM and 20 to 40 seconds per answer.

Sample images:
    Put your own images in a folder named "samples" next to this file and they
    become the default gallery. The file name is the label.
    Without that folder the app loads a few COCO validation images.
"""

import base64
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw
from transformers import Blip2Config, Blip2ForConditionalGeneration, Blip2Processor

MODEL_ID = "Salesforce/blip2-opt-2.7b"
CLIP_ID = "openai/clip-vit-base-patch32"

st.set_page_config(
    page_title="BLIP-2 Walkthrough · SEAS 8525",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS (house style: Lexend, navy titles, gold tags, light panels)
# ═════════════════════════════════════════════════════════════════════════════
NAVY = "#002147"
GOLD = "#FFC400"
BG = "#F3F6FB"
BORDER = "#D9E1EC"
INK = "#1E293B"
MUTED = "#64748B"

IMG, IMG_BG = "#1D4ED8", "#E3ECFF"        # image side = blue
TXT, TXT_BG = "#0E9F6E", "#DDF5EC"        # text side = green
PROJ, PROJ_BG = "#D97706", "#FFF2DB"      # projection = orange
SPACE, SPACE_BG = "#7C3AED", "#F1EAFE"    # shared space = purple
POS, POS_BG = "#0E9F6E", "#CFEFDF"        # matching pair
NEG, NEG_BG = "#E11D48", "#FDE2E8"        # non-matching pair

PAIR_COLORS = ["#1D4ED8", "#E11D48", "#D97706", "#0E9F6E",
               "#7C3AED", "#0891B2", "#BE185D", "#4B5563"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, p, li, label, input, textarea, button, h1, h2, h3, h4,
.stMarkdown, [data-testid="stWidgetLabel"], [data-testid="stExpander"] summary p {
    font-family: 'Lexend', system-ui, sans-serif !important;
}
code, pre, .stCode code { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stAppViewContainer"] { background: #F3F6FB; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1320px; }
footer, [data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #D9E1EC; }
p, li { font-size: 16px; color: #1E293B; line-height: 1.6; }
[data-testid="stExpander"] { background: #FFFFFF; border-radius: 12px; border-color: #D9E1EC; }

/* section banner */
.hero { background: #002147; border-radius: 16px; padding: 18px 28px 16px; margin: 4px 0 16px;
        display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.hero .badge { background: #FFC400; color: #002147; font-weight: 800; font-size: 18px;
        width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.hero .t { color: #FFFFFF; font-weight: 800; font-size: 32px; letter-spacing: -0.3px; }
.hero .s { color: #C9D6EA; font-size: 17px; flex-basis: 100%; margin-left: 56px; margin-top: -6px; }

/* takeaway box at the end of each section */
.key { background: #FFFFFF; border: 1px solid #D9E1EC; border-top: 6px solid #FFC400;
       border-radius: 16px; padding: 22px 36px; margin: 22px auto 10px; text-align: center;
       box-shadow: 0 4px 14px rgba(0, 33, 71, 0.08); }
.key .x { font-size: 21px; font-weight: 500; color: #002147; line-height: 1.6; max-width: 980px; margin: 0 auto; }
.key .x b { font-weight: 800; }

/* top navigation: big, clear section buttons */
.st-key-topnav { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 10px 12px;
                 box-shadow: 0 2px 10px rgba(0, 33, 71, 0.06); margin-bottom: 4px; }
.st-key-topnav [data-testid="stButtonGroup"], .st-key-topnav [data-testid="stButtonGroup"] > div { width: 100%; }
.st-key-topnav [data-testid="stButtonGroup"] button { flex: 1 1 0; min-height: 54px; padding: 8px 14px;
                 border-radius: 12px !important; border: 1.5px solid #D9E1EC; background: #F3F6FB; }
.st-key-topnav [data-testid="stButtonGroup"] button p { font-size: 18px !important; font-weight: 700 !important; color: #002147; }
.st-key-topnav [data-testid="stButtonGroup"] button:hover { border-color: #002147; }
.st-key-topnav [data-testid="stBaseButton-segmented_controlActive"] { background: #002147 !important; border-color: #002147 !important; }
.st-key-topnav [data-testid="stBaseButton-segmented_controlActive"] p { color: #FFC400 !important; }
.st-key-topnav [role="radiogroup"] label p { font-size: 18px !important; font-weight: 700; color: #002147; }

/* "by the numbers" fact cards */
.facts { display: grid; grid-template-columns: 1.35fr 1fr 1fr; gap: 14px; margin: 6px 0 4px; }
@media (max-width: 1000px) { .facts { grid-template-columns: 1fr; } }
.fgroup { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 14px 16px; }
.fgroup .gh { font-weight: 800; font-size: 16px; margin-bottom: 2px; }
.fgroup .gs { font-size: 13px; color: #64748B; margin-bottom: 10px; }
.fact { padding: 10px 0; border-top: 1px solid #EEF2F7; }
.fact .n { font-size: 32px; font-weight: 800; line-height: 1.1; white-space: nowrap; }
.fact .k { font-weight: 700; font-size: 15.5px; margin-top: 2px; color: #002147; }
.fact .d { font-size: 13.5px; color: #475569; line-height: 1.45; margin-top: 2px; }

/* small stat chips */
.chips { display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 12px; }
.chip { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 14px; padding: 10px 18px; min-width: 150px; }
.chip b { display: block; font-size: 24px; font-weight: 800; color: #002147; }
.chip span { font-size: 13px; color: #64748B; }

/* panels and pipelines */
.panel { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 18px 20px; margin: 6px 0 10px; }
.ptitle { font-weight: 700; color: #002147; font-size: 17px; margin-bottom: 10px; display: flex; gap: 10px; align-items: center; }
.tag { display: inline-block; border-radius: 999px; padding: 2px 12px; font-size: 13px; font-weight: 700; }
.pipe { display: flex; align-items: center; gap: 6px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 2px; }
.pipe > * { flex-shrink: 0; }
.arrow { font-size: 22px; color: #94A3B8; font-weight: 700; }
.node { border-radius: 12px; padding: 8px 11px; text-align: center; font-weight: 700; font-size: 14px; line-height: 1.3; }
.node small { display: block; font-weight: 500; color: #64748B; font-size: 12px; margin-top: 2px; }
.n-img  { background: #E3ECFF; border: 2px solid #1D4ED8; color: #1D4ED8; }
.n-txt  { background: #DDF5EC; border: 2px solid #0E9F6E; color: #0B7A55; }
.n-proj { background: #FFF2DB; border: 2px solid #D97706; color: #B45309; }
.n-space{ background: #F1EAFE; border: 2px solid #7C3AED; color: #6D28D9; }
.n-plain{ background: #FFFFFF; border: 1px dashed #94A3B8; color: #1E293B; }
.thumb { border-radius: 10px; display: block; }

/* token sequences */
.seq { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; justify-content: center; }
.seq img { width: 24px; height: 24px; border-radius: 4px; border: 1px solid #93B4F5; }
.tk { display: inline-block; border-radius: 8px; padding: 3px 7px; font-size: 13px; font-weight: 600;
      background: #FFFFFF; border: 1.5px solid #0E9F6E; color: #0B7A55; }
.tk.sp { background: #F1F5F9; border-color: #94A3B8; color: #475569; }
.tk.eot { background: #0E9F6E; border-color: #0E9F6E; color: #FFFFFF; }
.tk.cls { background: #1D4ED8; border-color: #1D4ED8; color: #FFFFFF; }

/* embedding strips */
.strip { display: flex; gap: 1px; }
.strip i { display: block; width: 8px; height: 30px; border-radius: 2px; }
.grid5 { display: grid; grid-template-columns: auto 34px auto 34px 1fr; gap: 12px 6px; align-items: center; }
.slabel { font-size: 12.5px; color: #64748B; margin-top: 4px; font-weight: 500; }

/* similarity matrix */
table.mx { border-collapse: separate; border-spacing: 6px; margin: 0 auto; }
table.mx td { width: 104px; height: 74px; text-align: center; border-radius: 10px; font-size: 19px;
              color: #1E293B; border: 1px solid rgba(0,0,0,0.06); }
table.mx td.d { font-weight: 800; }
table.mx td.win { box-shadow: inset 0 0 0 3px #002147; }
table.mx th { font-weight: 600; font-size: 14px; color: #1E293B; }
table.mx th.col span { display: block; border-radius: 10px; padding: 6px 8px; border: 2px solid; line-height: 1.25; }
table.mx th.row { text-align: left; }
table.mx th.row div { display: flex; align-items: center; gap: 8px; }
table.mx th.row img { width: 66px; height: 66px; object-fit: cover; border-radius: 10px; border: 3px solid; }

/* zero-shot step cards */
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 1000px) { .steps { grid-template-columns: 1fr 1fr; } }
.step { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 14px; }
.step h4 { margin: 0 0 12px 0; padding: 8px 12px; border-radius: 10px; font-size: 17px; font-weight: 700; }
.bar { display: grid; grid-template-columns: 1fr 62px; gap: 6px; align-items: center; margin: 5px 0; font-size: 14px; }
.bar .track { background: #F1F5F9; border-radius: 6px; height: 26px; position: relative; overflow: hidden; }
.bar .fill { height: 100%; border-radius: 6px; }
.bar .lab { position: absolute; left: 8px; top: 3px; font-weight: 600; color: #1E293B; white-space: nowrap; }
.bar .v { text-align: right; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 13.5px; }

/* sidebar gallery */
.gal { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.gal div { font-size: 11px; color: #475569; text-align: center; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.gal img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SMALL HTML HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def html(s: str):
    """Render an HTML snippet. Lines are joined so Markdown never sees
    indentation or blank lines (which would break the HTML)."""
    one_line = " ".join(line.strip() for line in s.splitlines() if line.strip())
    st.markdown(f"<div>{one_line}</div>", unsafe_allow_html=True)


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("$", "&#36;"))


def hero(num, title, subtitle):
    html(f'<div class="hero"><div class="badge">{num}</div><div class="t">{title}</div>'
         f'<div class="s">{subtitle}</div></div>')


def key_idea(text):
    html(f'<div class="key"><div class="x">{text}</div></div>')


def chips(items):
    inner = "".join(f'<div class="chip"><b>{v}</b><span>{k}</span></div>' for v, k in items)
    html(f'<div class="chips">{inner}</div>')


def mix(c1, c2, t):
    t = float(np.clip(t, 0, 1))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def val_color(t, pos_color):
    """t in [-1, 1]: positive values tint toward pos_color, negative toward red."""
    return mix("#FFFFFF", pos_color, t) if t >= 0 else mix("#FFFFFF", NEG, -t)


def strip_html(vec, color, n=64):
    v = np.asarray(vec[:n], dtype=float)
    m = max(float(np.abs(v).max()), 1e-6)
    cells = "".join(f'<i style="background:{val_color(x / m, color)}"></i>' for x in v)
    return f'<div class="strip">{cells}</div>'


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def seg(label, options, key, default=None):
    """Segmented control with a radio fallback for older Streamlit versions."""
    if key not in st.session_state:
        st.session_state[key] = default or options[0]
    if hasattr(st, "segmented_control"):
        val = st.segmented_control(label, options, key=key)
    else:
        val = st.radio(label, options, key=key, horizontal=True)
    return val or options[0]


def code_panel(title, code):
    with st.expander(f"💻  Code · {title}", expanded=False):
        st.code(code, language="python")


# ═════════════════════════════════════════════════════════════════════════════
# IMAGES
# ═════════════════════════════════════════════════════════════════════════════
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}

COCO = "http://images.cocodataset.org/val2017/"
SAMPLE_URLS = [
    ("cat", "two cats sleeping on a couch", COCO + "000000039769.jpg"),
    ("bear", "a brown bear sitting in the grass", COCO + "000000000285.jpg"),
    ("stop sign", "a red stop sign by the road", COCO + "000000000724.jpg"),
    ("skier", "a person skiing down a snowy slope", COCO + "000000000785.jpg"),
    ("teddy bear", "a group of teddy bears", COCO + "000000000776.jpg"),
    ("kitchen", "a kitchen with a refrigerator", COCO + "000000000802.jpg"),
    ("bedroom", "a bedroom with a bed", COCO + "000000000632.jpg"),
    ("living room", "a living room with a couch and a tv", COCO + "000000000139.jpg"),
]


@st.cache_data(show_spinner=False)
def fetch_bytes(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read()
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=128)
def to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def clip_crop(img, size=224):
    """Resize the short side to `size`, then center crop (what CLIPProcessor does)."""
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def pil_uri(img, fmt="JPEG", quality=85):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False, max_entries=256)
def thumb_uri(b: bytes, size=160):
    return pil_uri(clip_crop(to_pil(b), size))


def build_pool(uploads):
    """Ordered dict: label -> {"bytes", "caption"}."""
    pool = {}

    def add(label, caption, data):
        name, k = label, 2
        while name in pool:
            name = f"{label} ({k})"
            k += 1
        pool[name] = {"bytes": data, "caption": caption}

    local = Path(__file__).parent / "samples"
    files = sorted(p for p in local.glob("*") if p.suffix.lower() in IMG_EXT) if local.exists() else []
    if files:
        for p in files:
            lbl = p.stem.replace("_", " ")
            add(lbl, f"a photo of {article(lbl)} {lbl}", p.read_bytes())
    else:
        for lbl, cap, url in SAMPLE_URLS:
            data = fetch_bytes(url)
            if data:
                add(lbl, cap, data)

    for f in uploads or []:
        lbl = Path(f.name).stem.replace("_", " ")
        add(lbl, f"a photo of {article(lbl)} {lbl}", f.getvalue())
    return pool



BLIP_CSS = """
.stButton > button[kind="primary"] { background: #002147; border: none; color: #FFFFFF; font-weight: 700;
    font-size: 16px; padding: 10px 26px; border-radius: 12px; }
.stButton > button[kind="primary"]:hover { background: #0B3A6E; color: #FFC400; }
.stButton > button[kind="primary"] p { color: #FFFFFF; font-size: 16px; font-weight: 700; }
.vt { display: inline-block; width: 16px; height: 16px; border-radius: 4px; background: #C4B5FD;
      border: 1.5px solid #7C3AED; margin: 1px; vertical-align: middle; }
.vt.big { width: 22px; height: 22px; }
.tk.pr { background: #F1F5F9; border-color: #94A3B8; color: #334155; }
.tk.fmt { background: #FFF2DB; border-color: #D97706; color: #B45309; }
.tk.ans { background: #0E9F6E; border-color: #0E9F6E; color: #FFFFFF; }
.flag { display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 12.5px; font-weight: 700; }
.flag.fz { background: #E3ECFF; color: #1D4ED8; }
.flag.tr { background: #FFE7D1; color: #C2410C; }
.flag.na { background: #F1F5F9; color: #94A3B8; }
.stack { display: flex; flex-direction: column; gap: 8px; position: relative; padding: 12px 14px;
         border: 2px solid #D97706; border-radius: 14px; background: #FFF8EC; }
.blk { border-radius: 10px; padding: 8px 12px; font-weight: 700; font-size: 14px; text-align: left; }
.blk small { display: block; font-weight: 500; color: #475569; font-size: 12.5px; }
.out { background: #FFFFFF; border: 1.5px solid #0E9F6E; border-left: 6px solid #0E9F6E; border-radius: 12px;
       padding: 12px 16px; margin: 8px 0; font-size: 18px; color: #0F3D2E; }
.out small { display: block; color: #64748B; font-size: 12.5px; font-weight: 600; margin-bottom: 2px; }
.cards3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
@media (max-width: 1000px) { .cards3 { grid-template-columns: 1fr; } }
.tiles { display: grid; grid-template-columns: repeat(8, 1fr); gap: 6px; }
.tiles div { text-align: center; font-size: 11px; color: #475569; font-weight: 600; }
.tiles img { width: 100%; border-radius: 6px; display: block; }
table.cmp { width: 100%; border-collapse: separate; border-spacing: 0; background: #FFFFFF;
            border: 1px solid #D9E1EC; border-radius: 14px; overflow: hidden; }
table.cmp th, table.cmp td { padding: 10px 14px; text-align: left; font-size: 15px; border-bottom: 1px solid #EEF2F7; }
table.cmp th { font-size: 16px; }
table.cmp td:first-child { color: #64748B; font-weight: 600; width: 22%; }
.pbar { display: flex; height: 42px; border-radius: 12px; overflow: hidden; border: 1px solid #D9E1EC; }
.pbar div { display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-weight: 700; font-size: 14px; }
"""
st.markdown(f"<style>{BLIP_CSS}</style>", unsafe_allow_html=True)


def facts_panel(facts, note):
    html(f'<div style="font-weight:800;font-size:20px;color:{NAVY};margin:14px 0 6px">{note[0]} '
         f'<span style="font-weight:500;font-size:15px;color:{MUTED}">· {note[1]}</span></div>')
    cols = ""
    for title, sub, color, items in facts:
        rows = "".join(f'<div class="fact"><div class="n" style="color:{color}">{n}</div>'
                       f'<div><div class="k">{k}</div><div class="d">{esc(d)}</div></div></div>'
                       for n, k, d in items)
        cols += (f'<div class="fgroup" style="border-top:5px solid {color}"><div class="gh" style="color:{color}">{title}</div>'
                 f'<div class="gs">{sub}</div>{rows}</div>')
    html(f'<div class="facts">{cols}</div>')


def vtokens(n=8, total=32, big=False):
    sq = "".join(f'<span class="vt{" big" if big else ""}"></span>' for _ in range(n))
    more = f'<span style="color:{MUTED};font-weight:700;margin-left:4px">··· ×{total}</span>' if total > n else ""
    return sq + more


def wrap(text, width, max_lines):
    lines, cur = [], ""
    for w in str(text).split():
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(1, width - 1)] + "…"
    return lines or [""]


def grid_overlay(img, n):
    g = img.copy()
    d = ImageDraw.Draw(g)
    w = img.size[0]
    for k in range(n + 1):
        x = round(k * w / n)
        d.line([(x, 0), (x, w)], fill="white", width=1)
        d.line([(0, x), (w, x)], fill="white", width=1)
    return g


def patches(img, n):
    w = img.size[0]
    p = w / n
    return [img.crop((round(c * p), round(r * p), round((c + 1) * p), round((r + 1) * p)))
            for r in range(n) for c in range(n)]


# ═════════════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════════════
FALLBACK = dict(vis_hidden=1408, vis_layers=39, patch=14, image=224, q_hidden=768, q_layers=12, q_heads=12,
                cross_freq=2, n_query=32, llm_hidden=2560, llm_layers=32, llm_vocab=50272)


@st.cache_resource(show_spinner=False)
def arch():
    """Architecture numbers from the model's config.json only (a few KB, no weights)."""
    try:
        cfg = Blip2Config.from_pretrained(MODEL_ID)
        vc, qc, tc = cfg.vision_config, cfg.qformer_config, cfg.text_config
        a = dict(vis_hidden=vc.hidden_size, vis_layers=vc.num_hidden_layers, patch=vc.patch_size,
                 image=vc.image_size, q_hidden=qc.hidden_size, q_layers=qc.num_hidden_layers,
                 q_heads=qc.num_attention_heads, cross_freq=qc.cross_attention_frequency,
                 n_query=cfg.num_query_tokens, llm_hidden=tc.hidden_size, llm_layers=tc.num_hidden_layers,
                 llm_vocab=tc.vocab_size)
    except Exception:
        a = dict(FALLBACK)
    a["side"] = a["image"] // a["patch"]
    a["n_tok"] = a["side"] ** 2 + 1
    return a


@st.cache_resource(show_spinner="Loading BLIP-2 (first run downloads the weights, several GB)...")
def load_blip2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, processor, device


def model_loaded():
    return st.session_state.get("_blip_loaded", False)


def get_blip():
    out = load_blip2()
    st.session_state["_blip_loaded"] = True
    return out


def _clean(text, prompt):
    t = text.strip()
    if prompt:
        p = prompt.strip()
        if t.startswith(p):
            t = t[len(p):]
        if "Answer:" in t:
            t = t.split("Answer:")[-1]
    return t.strip() or "(no answer)"


def blip_generate(img_bytes, prompt=None, max_new_tokens=40, do_sample=False, temperature=1.0, n=1):
    model, processor, device = get_blip()
    dtype = torch.float16 if device == "cuda" else torch.float32
    kw = dict(images=to_pil(img_bytes), return_tensors="pt")
    if prompt:
        kw["text"] = prompt
    inputs = processor(**kw).to(device, dtype)
    gen = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        gen.update(do_sample=True, temperature=temperature, top_p=0.9, num_return_sequences=n)
    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    return [_clean(t, prompt) for t in processor.batch_decode(out, skip_special_tokens=True)]


def query_attention(img_bytes):
    """Cross-attention of the 32 queries over the image patches, averaged over heads and layers."""
    model, processor, device = get_blip()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pv = processor(images=to_pil(img_bytes), return_tensors="pt")["pixel_values"].to(device, dtype)
    with torch.no_grad():
        img_emb = model.vision_model(pixel_values=pv).last_hidden_state
        q = model.query_tokens.expand(img_emb.shape[0], -1, -1)
        att = torch.ones(img_emb.shape[:-1], dtype=torch.long, device=img_emb.device)
        out = model.qformer(query_embeds=q, encoder_hidden_states=img_emb, encoder_attention_mask=att,
                            output_attentions=True, return_dict=True)
    ca = [a for a in (getattr(out, "cross_attentions", None) or []) if a is not None]
    if not ca:
        raise RuntimeError("this transformers version does not return cross-attention weights")
    A = torch.stack([a.float().mean(1) for a in ca]).mean(0)[0]       # [32, 257]
    A = A[:, 1:].cpu().numpy()                                          # drop the CLS column
    side = int(round(np.sqrt(A.shape[1])))
    return A.reshape(A.shape[0], side, side)


@st.cache_resource(show_spinner="Loading CLIP for the comparison...")
def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(CLIP_ID).eval()
    p = CLIPProcessor.from_pretrained(CLIP_ID)
    return m, p


def _unit(x):
    if not isinstance(x, torch.Tensor):
        for name in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(x, name, None)
            if v is not None:
                x = v
                break
    return x / x.norm(dim=-1, keepdim=True)


def clip_scores(img_bytes, prompts):
    m, p = load_clip()
    with torch.no_grad():
        i = _unit(m.get_image_features(**p(images=[to_pil(img_bytes)], return_tensors="pt")))
        t = _unit(m.get_text_features(**p(text=prompts, return_tensors="pt", padding=True)))
        scale = float(m.logit_scale.exp().item())
    s = (i @ t.T)[0].cpu().numpy()
    return s, softmax_np(s * scale)


def softmax_np(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def run_once(key, sig, fn):
    """Run fn() when the button was pressed; keep its result while the inputs stay the same."""
    try:
        st.session_state[key] = {"sig": sig, "val": fn()}
    except Exception as e:  # show the problem instead of crashing the page
        st.session_state[key] = {"sig": sig, "err": f"{type(e).__name__}: {e}"}


def result(key, sig):
    r = st.session_state.get(key)
    if r and r["sig"] == sig:
        if "err" in r:
            st.error(f"The model could not run here: {r['err']}")
            return None
        return r["val"]
    return None


def run_button(label, key):
    return st.button(f"▶  {label}", key=key, type="primary")


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    html(f'<div style="font-weight:800;font-size:22px;color:{NAVY}">🧠 BLIP-2 Walkthrough</div>'
         f'<div style="margin-top:4px"><span class="tag" style="background:{GOLD};color:{NAVY}">SEAS 8525</span>'
         f'<span style="color:{MUTED};font-size:14px;margin-left:8px">Week 6 · Dr. Elbasheer</span></div>')
    st.divider()
    st.markdown("**Image gallery**")
    st.caption("Every section picks images from here. Add your own below.")
    uploads = st.file_uploader("Add images", type=["png", "jpg", "jpeg", "webp"],
                               accept_multiple_files=True, label_visibility="collapsed")
    POOL = build_pool(uploads)
    if POOL:
        tiles = "".join(f'<div><img src="{thumb_uri(v["bytes"], 120)}"/>{esc(k)}</div>'
                        for k, v in POOL.items())
        html(f'<div class="gal">{tiles}</div>')
    else:
        st.warning("No sample images could be loaded. Upload a few images above.")
    st.divider()
    if model_loaded():
        st.caption("✅ BLIP-2 is loaded.")
    else:
        st.caption("The model loads the first time you press a ▶ button. The diagrams work without it.")
    if torch.cuda.is_available():
        st.caption(f"⚡ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.caption("Running on CPU: each generation takes 20 to 40 seconds and needs about 16 GB of RAM.")
    with st.expander("Run it locally"):
        st.code("pip install -r requirements.txt\nstreamlit run blip2_app.py", language="bash")

LABELS = list(POOL)
A = arch()


def need_images(k=1):
    if len(LABELS) < k:
        st.info(f"This section needs at least {k} image(s). Add some in the sidebar gallery.")
        return True
    return False


def pick_image(key, label="Image", index=0):
    return st.selectbox(label, LABELS, index=min(index, len(LABELS) - 1), key=key)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 · BIG PICTURE
# ═════════════════════════════════════════════════════════════════════════════
def arch_svg(img_uri, out_text):
    AR = 'stroke="#334155" stroke-width="2.4" marker-end="url(#ah)"'
    cy = 170

    def arrow(x1, x2, y=cy):
        return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" {AR}/>'

    def squares(x, y, n, fill, stroke, s=14, g=4):
        return "".join(f'<rect x="{x + k * (s + g)}" y="{y}" width="{s}" height="{s}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
                       for k in range(n))

    stack = "".join(f'<rect x="{334 + c * 26}" y="{118 + r * 22}" width="20" height="17" rx="3" fill="{IMG_BG}" stroke="{IMG}" stroke-width="1.5"/>'
                    for r in range(5) for c in range(2))
    out_lines = wrap(out_text, 13, 4)
    out_txt = "".join(f'<text x="1186" y="{cy - 8 * (len(out_lines) - 1) + 16 * k + 4}" text-anchor="middle" font-size="12.5" fill="#1E293B">{esc(l)}</text>'
                      for k, l in enumerate(out_lines))
    return f"""
    <div class="panel">
    <svg viewBox="0 52 1240 324" width="100%" style="font-family:Lexend, sans-serif">
      <defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#334155"/></marker></defs>

      <text x="75" y="98" text-anchor="middle" font-size="16" font-weight="700" fill="{IMG}">Image</text>
      <image href="{img_uri}" x="20" y="115" width="110" height="110" preserveAspectRatio="xMidYMid slice"/>
      <rect x="20" y="115" width="110" height="110" fill="none" stroke="{IMG}" stroke-width="3" rx="6"/>
      <text x="75" y="246" text-anchor="middle" font-size="13" fill="#475569">{A['image']} × {A['image']}</text>
      {arrow(134, 156)}

      <polygon points="162,80 302,115 302,225 162,260" fill="{IMG_BG}" stroke="{IMG}" stroke-width="2.5"/>
      <text x="232" y="158" text-anchor="middle" font-size="16" font-weight="700" fill="{IMG}">Image Encoder</text>
      <text x="232" y="180" text-anchor="middle" font-size="13.5" fill="#475569">ViT-g/14</text>
      <text x="232" y="205" text-anchor="middle" font-size="13.5" font-weight="700" fill="{IMG}">❄ frozen</text>
      {arrow(306, 326)}

      {stack}
      <text x="357" y="249" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">{A['n_tok']} × {A['vis_hidden']}</text>
      <text x="357" y="266" text-anchor="middle" font-size="12" fill="#475569">image features</text>
      {arrow(388, 412)}

      <rect x="418" y="66" width="194" height="210" rx="14" fill="{PROJ_BG}" stroke="{PROJ}" stroke-width="2.5"/>
      <text x="515" y="94" text-anchor="middle" font-size="17" font-weight="800" fill="#B45309">Q-Former 🔥</text>
      <text x="515" y="113" text-anchor="middle" font-size="12.5" fill="#B45309">trainable bridge</text>
      {squares(441, 124, 8, "#C4B5FD", SPACE)}
      <text x="515" y="158" text-anchor="middle" font-size="12" fill="#6D28D9">{A['n_query']} learned queries</text>
      <rect x="436" y="168" width="158" height="26" rx="7" fill="{SPACE_BG}" stroke="{SPACE}" stroke-width="1.2"/>
      <text x="515" y="186" text-anchor="middle" font-size="12.5" fill="#6D28D9">self-attention</text>
      <rect x="436" y="200" width="158" height="26" rx="7" fill="#FFE4D6" stroke="#C2410C" stroke-width="1.2"/>
      <text x="515" y="218" text-anchor="middle" font-size="12.5" fill="#9A3412">cross-attention → image</text>
      <rect x="436" y="232" width="158" height="26" rx="7" fill="{IMG_BG}" stroke="{IMG}" stroke-width="1.2"/>
      <text x="515" y="250" text-anchor="middle" font-size="12.5" fill="{IMG}">feed-forward</text>
      {arrow(616, 638)}

      {squares(644, cy - 7, 4, "#C4B5FD", SPACE)}
      <text x="679" y="{cy + 34}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">{A['n_query']} × {A['q_hidden']}</text>
      <text x="679" y="{cy + 51}" text-anchor="middle" font-size="12" fill="#475569">visual tokens</text>
      {arrow(718, 740)}

      <rect x="746" y="{cy - 38}" width="108" height="76" rx="12" fill="{NEG_BG}" stroke="{NEG}" stroke-width="2.5"/>
      <text x="800" y="{cy - 9}" text-anchor="middle" font-size="15" font-weight="700" fill="#BE123C">Projection 🔥</text>
      <text x="800" y="{cy + 15}" text-anchor="middle" font-size="13" fill="#475569">{A['q_hidden']} → {A['llm_hidden']}</text>
      {arrow(858, 880)}

      {squares(886, cy - 7, 4, "#C4B5FD", SPACE)}
      <text x="921" y="{cy + 34}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">{A['n_query']} × {A['llm_hidden']}</text>
      <text x="921" y="{cy + 51}" text-anchor="middle" font-size="12" fill="#475569">soft prompt</text>
      {arrow(960, 982)}

      <polygon points="988,115 1118,80 1118,260 988,225" fill="{TXT_BG}" stroke="{TXT}" stroke-width="2.5"/>
      <text x="1053" y="158" text-anchor="middle" font-size="16" font-weight="700" fill="#0B7A55">Frozen LLM</text>
      <text x="1053" y="180" text-anchor="middle" font-size="13.5" fill="#475569">OPT-2.7B</text>
      <text x="1053" y="205" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0B7A55">❄ frozen</text>
      {arrow(1122, 1132)}
      <rect x="1136" y="{cy - 46}" width="100" height="92" rx="10" fill="#FFFFFF" stroke="#94A3B8" stroke-width="1.5"/>
      {out_txt}
      <text x="1186" y="{cy - 56}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">Text out</text>

      <rect x="996" y="318" width="114" height="44" rx="10" fill="#FFFFFF" stroke="{TXT}" stroke-width="2" stroke-dasharray="6 4"/>
      <text x="1053" y="336" text-anchor="middle" font-size="13" fill="#0B7A55" font-weight="700">text prompt</text>
      <text x="1053" y="353" text-anchor="middle" font-size="12" fill="#475569">(optional)</text>
      <line x1="1053" y1="316" x2="1053" y2="266" {AR}/>

      <text x="20" y="335" font-size="14" font-weight="700" fill="{IMG}">❄ frozen: never updated</text>
      <text x="20" y="360" font-size="14" font-weight="700" fill="#C2410C">🔥 trainable: the only parts that learn</text>
    </svg></div>"""


PARAMS = [("Image encoder · ViT-g/14", 1.0, IMG, "❄ frozen"),
          ("Q-Former + projection", 0.107, PROJ, "🔥 trained"),
          ("LLM · OPT-2.7B", 2.7, TXT, "❄ frozen")]


def fmt_b(b):
    return f"{b * 1000:.0f}M" if b < 1 else f"{b:g}B"


def param_bar():
    total = sum(p[1] for p in PARAMS)
    segs, legend = "", ""
    for name, b, c, tag in PARAMS:
        w = max(b / total * 100, 5)
        segs += f'<div style="width:{w:.1f}%;background:{c}">{"" if w < 8 else f"{b:g}B"}</div>'
        legend += (f'<div style="display:flex;gap:8px;align-items:center"><span style="width:14px;height:14px;'
                   f'border-radius:4px;background:{c};display:inline-block"></span><b style="color:{NAVY}">{name}</b>'
                   f'<span style="color:{MUTED}">{fmt_b(b)} · {tag}</span></div>')
    pct = PARAMS[1][1] / total * 100
    html(f"""<div class="panel">
      <div class="ptitle">What actually gets trained?
        <span class="tag" style="background:#FFE7D1;color:#C2410C">only {pct:.0f}% of the parameters</span></div>
      <div class="pbar">{segs}</div>
      <div style="display:flex;gap:26px;flex-wrap:wrap;margin-top:10px;font-size:14px">{legend}</div>
    </div>""")


FACTS = [
    ("Training setup", "What the original authors used", PROJ, [
        ("129M", "pre-training images",
         "Image-text pairs from COCO, Visual Genome, CC3M, CC12M, SBU, and 115M images from LAION-400M."),
        ("2", "training stages", "Stage 1 teaches the Q-Former to see. Stage 2 teaches it to talk to the LLM."),
        ("188M", "Q-Former parameters (stage 1)",
         "Started from BERT-base. Includes a text branch that is only needed during stage 1."),
    ]),
    ("This model", "The checkpoint running in this app", IMG, [
        ("ViT-g/14", "frozen image encoder", "From EVA-CLIP, about 1B parameters, 14×14 pixel patches."),
        ("32 × 768", "learned query tokens", "Trainable vectors that ask the image what matters for language."),
        ("OPT-2.7B", "frozen language model", "Meta's open LLM with hidden size 2560. It never sees a gradient."),
    ]),
    ("Results", "Reported in the paper", POS, [
        ("+8.7%", "over Flamingo-80B", "On zero-shot VQAv2, with the largest BLIP-2 (ViT-g + FlanT5-XXL)."),
        ("54×", "fewer trainable parameters", "Than Flamingo-80B, because both giant backbones stay frozen."),
    ]),
]


def sec_big_picture():
    hero(1, "BLIP-2: a bridge between two frozen giants",
         "Bootstrapping Language-Image Pre-training · Li et al., Salesforce 2023")
    if need_images():
        return
    c1, _ = st.columns([1, 2])
    with c1:
        lbl = pick_image("s1_img", "Example image")
    html(arch_svg(thumb_uri(POOL[lbl]["bytes"], 220), POOL[lbl]["caption"]))
    param_bar()
    facts_panel(FACTS, ("BLIP-2 by the numbers", "from the original paper, Li et al. 2023"))
    key_idea("Retraining a 1B vision model or a 2.7B language model is expensive, so BLIP-2 <b>freezes both</b>. "
             f"A small <b style='color:#C2410C'>trainable Q-Former</b> learns to turn the image into "
             f"<b style='color:{SPACE}'>{A['n_query']} tokens</b> that the frozen LLM reads like extra words.")

    with st.expander("🔍  Live parameter count (loads the model)"):
        if model_loaded() or st.button("Count parameters", key="s1_count"):
            model, _, _ = get_blip()
            parts = {"Image encoder (frozen)": model.vision_model,
                     "Q-Former + queries (trained)": model.qformer,
                     "Projection (trained)": model.language_projection,
                     "LLM (frozen)": model.language_model}
            counts = {k: sum(p.numel() for p in v.parameters()) for k, v in parts.items()}
            counts["Q-Former + queries (trained)"] += model.query_tokens.numel()
            st.table(pd.DataFrame({"component": list(counts),
                                   "parameters": [f"{v / 1e6:,.0f}M" for v in counts.values()]}))
            st.caption("The paper's 188M counts the full stage 1 Q-Former, including its text branch. "
                       "This inference checkpoint keeps only the parts used for generation.")

    code_panel("load BLIP-2 and caption an image", f'''\
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

processor = Blip2Processor.from_pretrained("{MODEL_ID}")
model = Blip2ForConditionalGeneration.from_pretrained(
    "{MODEL_ID}",
    torch_dtype=torch.float16,   # use torch.float32 on CPU
    device_map="auto",
).eval()

inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
out = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(out[0], skip_special_tokens=True))

# Sizes: ViT-g/14 about 1B (frozen) · Q-Former + projection about 107M (trained)
#        OPT-2.7B about 2.7B (frozen) · total about 3.8B''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 · INSIDE THE Q-FORMER
# ═════════════════════════════════════════════════════════════════════════════
def attn_tile(img224, a, size=104):
    a = (a - a.min()) / (np.ptp(a) + 1e-9)
    up = np.asarray(Image.fromarray(np.uint8(a * 255)).resize(img224.size, Image.BILINEAR), dtype=float) / 255
    base = np.asarray(img224, dtype=float)
    tint = np.array([255, 196, 0], dtype=float)
    out = base * (0.18 + 0.82 * up[..., None]) * (1 - 0.25 * up[..., None]) + tint * 0.25 * up[..., None]
    return Image.fromarray(np.uint8(np.clip(out, 0, 255))).resize((size, size))


def sec_qformer():
    hero(2, "Inside the Q-Former",
         f"From {A['n_tok']} × {A['vis_hidden']} image features to {A['n_query']} language-ready vectors")
    if need_images():
        return
    lbl = pick_image("s2_img")
    b = POOL[lbl]["bytes"]
    side = A["side"]
    img224 = clip_crop(to_pil(b), A["image"])
    ps = patches(img224, side)
    seq = ('<span class="tk cls">CLS</span>' + "".join(f'<img src="{pil_uri(q)}"/>' for q in ps[:4]) +
           '<span style="color:#64748B;font-weight:700">···</span>' + f'<img src="{pil_uri(ps[-1])}"/>')

    html(f"""
    <div class="panel">
      <div class="ptitle"><span class="tag" style="background:{IMG_BG};color:{IMG}">1 · FROZEN</span>The image encoder turns the photo into features</div>
      <div class="pipe">
        <div class="node n-plain"><img class="thumb" src="{pil_uri(grid_overlay(img224, side))}" width="128"/>
          <small>{A['image']}×{A['image']} → {side}×{side} = {side * side} patches</small></div>
        <div class="arrow">→</div>
        <div class="node n-img" style="width:180px"><div class="seq">{seq}</div>
          <small>{A['n_tok']} tokens = [CLS] + {side * side} patches of {A['patch']}×{A['patch']}</small></div>
        <div class="arrow">→</div>
        <div class="node n-img">ViT-g/14 ❄<small>{A['vis_layers']} layers · width {A['vis_hidden']}</small></div>
        <div class="arrow">→</div>
        <div class="node n-img" style="font-size:18px">{A['n_tok']} × {A['vis_hidden']}<small>image features</small></div>
      </div>
    </div>""")

    queries = "".join('<span class="vt"></span>' for _ in range(A["n_query"]))
    outgrid = "".join('<span class="vt" style="background:#7C3AED;border-color:#5B21B6"></span>' for _ in range(A["n_query"]))
    html(f"""
    <div class="panel">
      <div class="ptitle"><span class="tag" style="background:#FFE7D1;color:#C2410C">2 · TRAINABLE 🔥</span>The Q-Former asks the image questions</div>
      <div class="pipe" style="align-items:stretch">
        <div class="node n-space" style="width:170px;display:flex;flex-direction:column;justify-content:center">
          <div style="line-height:0">{queries}</div>
          <div style="margin-top:8px">{A['n_query']} learnable queries</div>
          <small>{A['q_hidden']}-d each · not patches, not words</small></div>
        <div class="arrow" style="align-self:center">→</div>
        <div class="stack" style="width:330px">
          <div class="blk" style="background:{SPACE_BG};border:1.5px solid {SPACE};color:#6D28D9">A · Self-attention
            <small>the queries share information with each other</small></div>
          <div class="blk" style="background:#FFE4D6;border:1.5px solid #C2410C;color:#9A3412">B · Cross-attention to the image
            <small>Q = the {A['n_query']} queries · K, V = the {A['n_tok']} image features</small></div>
          <div class="blk" style="background:{IMG_BG};border:1.5px solid {IMG};color:{IMG}">C · Feed-forward
            <small>refine each query on its own</small></div>
          <div style="text-align:center;font-weight:800;color:#B45309;font-size:14px">× {A['q_layers']} blocks
            <span style="font-weight:500;color:#64748B">(cross-attention in every {A['cross_freq']}nd block)</span></div>
        </div>
        <div class="arrow" style="align-self:center">→</div>
        <div class="node n-space" style="width:170px;display:flex;flex-direction:column;justify-content:center">
          <div style="line-height:0">{outgrid}</div>
          <div style="margin-top:8px">{A['n_query']} × {A['q_hidden']}</div>
          <small>compact visual tokens, ready for the LLM</small></div>
      </div>
    </div>""")

    big = A["n_tok"] * A["vis_hidden"]
    small = A["n_query"] * A["q_hidden"]
    html(f"""
    <div class="panel">
      <div class="ptitle">How much smaller?</div>
      <div class="bar"><div class="track" style="height:32px"><div class="fill" style="width:100%;background:{IMG_BG}"></div>
        <span class="lab" style="top:6px">image features · {A['n_tok']} × {A['vis_hidden']}</span></div><div class="v">{big:,}</div></div>
      <div class="bar"><div class="track" style="height:32px"><div class="fill" style="width:{small / big * 100:.1f}%;background:#C4B5FD"></div>
        <span class="lab" style="top:6px">Q-Former output · {A['n_query']} × {A['q_hidden']}</span></div><div class="v">{small:,}</div></div>
      <div class="slabel" style="font-size:14px">About <b>{big / small:.0f}× fewer numbers</b>, and the LLM reads
        <b>{A['n_query']} tokens instead of {A['n_tok']}</b>.</div>
    </div>""")

    st.markdown(f"<div style='font-weight:700;color:{NAVY};font-size:17px;margin-top:6px'>"
                "Where does each query look?</div>", unsafe_allow_html=True)
    sig = ("attn", lbl)
    if run_button("Show the cross-attention of all 32 queries", "s2_run"):
        with st.spinner("Running the image encoder and the Q-Former..."):
            run_once("s2_res", sig, lambda: query_attention(b))
    maps = result("s2_res", sig)
    if maps is not None:
        tiles = "".join(f'<div><img src="{pil_uri(attn_tile(img224, maps[k]))}"/>q{k + 1}</div>'
                        for k in range(maps.shape[0]))
        html(f'<div class="panel"><div class="tiles">{tiles}</div>'
             f'<div class="slabel" style="text-align:center;font-size:14px;margin-top:8px">Bright = where that query '
             f'pulls information from (averaged over heads and cross-attention layers)</div></div>')
    else:
        st.caption("Press the button to run the model. Each tile will show the image regions one query attends to.")

    key_idea(f"The {A['n_query']} queries are <b>learned questions</b>. Through cross-attention they read the "
             f"{A['n_tok']} image features and keep only what is useful for language, so the LLM gets a "
             f"<b style='color:{SPACE}'>short, fixed-size summary</b> of any image.")

    code_panel("one forward pass, step by step", f'''\
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# 1. Frozen ViT-g/14
image_embeds = model.vision_model(pixel_values).last_hidden_state   # [1, {A['n_tok']}, {A['vis_hidden']}]

# 2. Q-Former: {A['n_query']} learned queries cross-attend to the image
query_tokens = model.query_tokens.expand(1, -1, -1)                  # [1, {A['n_query']}, {A['q_hidden']}]
q_out = model.qformer(
    query_embeds=query_tokens,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=torch.ones(image_embeds.shape[:-1], dtype=torch.long),
    output_attentions=True,                 # gives the maps shown above
).last_hidden_state                         # [1, {A['n_query']}, {A['q_hidden']}]

# 3. Linear projection into the LLM's embedding space
soft_prompt = model.language_projection(q_out)                       # [1, {A['n_query']}, {A['llm_hidden']}]''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 · TWO-STAGE TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def mask_svg(kind):
    rows = ["Q1", "Q2", "Q3", "a", "dog", "runs"]
    nq = 3
    n = len(rows)
    cell, L, T = 24, 44, 36

    def allowed(i, j):
        if kind == "itc":
            return (i < nq) == (j < nq)
        if kind == "itm":
            return True
        if i < nq:                       # ITG: queries see only queries
            return j < nq
        return j < nq or j <= i          # text sees all queries and earlier words

    out = [f'<svg viewBox="0 0 {L + n * cell + 4} {T + n * cell + 4}" width="190" style="font-family:Lexend">']
    out.append(f'<text x="{L + n * cell / 2}" y="12" text-anchor="middle" font-size="10.5" fill="#475569">attends to →</text>')
    for j, t in enumerate(rows):
        out.append(f'<text x="{L + j * cell + cell / 2 - 1}" y="{T - 6}" text-anchor="middle" font-size="10" '
                   f'fill="{SPACE if j < nq else "#0B7A55"}" font-weight="700">{t}</text>')
    for i, t in enumerate(rows):
        out.append(f'<text x="{L - 6}" y="{T + i * cell + 16}" text-anchor="end" font-size="10" '
                   f'fill="{SPACE if i < nq else "#0B7A55"}" font-weight="700">{t}</text>')
        for j in range(n):
            ok = allowed(i, j)
            fill = ("#A78BFA" if j < nq else "#5CC9A1") if ok else "#EEF2F7"
            out.append(f'<rect x="{L + j * cell}" y="{T + i * cell}" width="{cell - 3}" height="{cell - 3}" rx="4" fill="{fill}"/>')
    out.append("</svg>")
    return "".join(out)


def sec_training():
    hero(3, "Two-stage training", "First teach the Q-Former to see, then teach it to talk to the LLM")
    if need_images():
        return
    lbl = LABELS[0]
    thumb = thumb_uri(POOL[lbl]["bytes"], 120)
    cap = POOL[lbl]["caption"]

    objectives = [
        ("Image-Text Contrastive", "ITC", IMG, IMG_BG, "itc",
         "Pull matching image and text together, push others apart. Same idea as CLIP.",
         "Queries and text cannot see each other, so each side is summarized on its own."),
        ("Image-Text Matching", "ITM", TXT, TXT_BG, "itm",
         "A yes/no classifier: does this text really describe this image? Uses hard negatives.",
         "Everything sees everything, so the queries can compare fine details with the words."),
        ("Image-Grounded Text Generation", "ITG", SPACE, SPACE_BG, "itg",
         "Write the caption word by word, using only what the queries extracted.",
         "Text sees all queries but only earlier words. Queries never see the text."),
    ]
    cards = ""
    for title, short, c, bg, kind, what, mask_note in objectives:
        cards += f"""
        <div class="step" style="border-top:5px solid {c}">
          <h4 style="background:{bg};color:{c};min-height:58px;display:flex;align-items:center">{title} ({short})</h4>
          <div style="font-size:14.5px;color:#1E293B;min-height:44px">{what}</div>
          <div style="text-align:center;margin-top:8px">{mask_svg(kind)}</div>
          <div class="slabel" style="text-align:center">{mask_note}</div>
        </div>"""

    html(f"""
    <div class="panel" style="border-top:6px solid {IMG}">
      <div class="ptitle"><span class="badge" style="background:{IMG};color:#fff;width:30px;height:30px;border-radius:50%;
        display:inline-flex;align-items:center;justify-content:center">1</span>
        Stage 1 · Vision-language representation learning
        <span class="tag" style="background:#F1F5F9;color:#475569">no LLM involved</span></div>
      <div class="pipe" style="margin-bottom:14px">
        <img class="thumb" src="{thumb}" width="70"/>
        <div class="arrow">→</div>
        <div class="node n-img">Image encoder<small>❄ frozen</small></div>
        <div class="arrow">→</div>
        <div class="node" style="background:{PROJ_BG};border:2px solid {PROJ};color:#B45309">Q-Former 🔥<small>queries + text branch</small></div>
        <div class="arrow">←</div>
        <div class="node n-plain" style="font-weight:500;font-size:13.5px">"{esc(cap)}"</div>
        <div class="arrow">⇒</div>
        <div class="node n-plain">3 losses, trained together<small>L = L<sub>ITC</sub> + L<sub>ITM</sub> + L<sub>ITG</sub></small></div>
      </div>
      <div class="cards3">{cards}</div>
      <div class="slabel" style="text-align:center;font-size:13.5px;margin-top:8px">
        Grids show the self-attention mask: a colored cell means the row token may look at the column token.
        <b style="color:{SPACE}">Q</b> = query, <b style="color:#0B7A55">green</b> = text.</div>
    </div>""")

    html(f"""
    <div class="panel" style="border-top:6px solid {TXT}">
      <div class="ptitle"><span class="badge" style="background:{TXT};color:#fff;width:30px;height:30px;border-radius:50%;
        display:inline-flex;align-items:center;justify-content:center">2</span>
        Stage 2 · Vision-to-language generative learning
        <span class="tag" style="background:#F1F5F9;color:#475569">connect the frozen LLM</span></div>
      <div class="pipe">
        <img class="thumb" src="{thumb}" width="70"/>
        <div class="arrow">→</div>
        <div class="node n-img">Image encoder<small>❄ frozen</small></div>
        <div class="arrow">→</div>
        <div class="node" style="background:{PROJ_BG};border:2px solid {PROJ};color:#B45309">Q-Former 🔥<small>{A['n_query']} × {A['q_hidden']}</small></div>
        <div class="arrow">→</div>
        <div class="node" style="background:{NEG_BG};border:2px solid {NEG};color:#BE123C">Projection 🔥<small>{A['q_hidden']} → {A['llm_hidden']}</small></div>
        <div class="arrow">→</div>
        <div class="node n-txt">OPT-2.7B<small>❄ frozen</small></div>
        <div class="arrow">→</div>
        <div class="node n-plain" style="font-weight:500;font-size:13.5px">"{esc(cap)}"</div>
      </div>
      <div class="slabel" style="font-size:14px;margin-top:10px">Loss: ordinary next-word prediction on the caption.
        The gradient flows back <b>through</b> the frozen LLM, but only the Q-Former and projection change.</div>
    </div>""")

    rows = [("Image encoder (ViT-g/14)", "fz", "fz"), ("Q-Former", "tr", "tr"),
            ("Projection", "na", "tr"), ("LLM (OPT-2.7B)", "na", "fz")]
    name = {"fz": "❄ frozen", "tr": "🔥 trained", "na": "not used"}
    body = "".join(f'<tr><td>{r}</td><td><span class="flag {a}">{name[a]}</span></td>'
                   f'<td><span class="flag {b2}">{name[b2]}</span></td></tr>' for r, a, b2 in rows)
    html(f'<table class="cmp"><tr><th></th><th style="color:{IMG}">Stage 1</th><th style="color:#0B7A55">Stage 2</th></tr>{body}</table>')

    key_idea("Stage 1 teaches the queries <b>what to look for</b> using three objectives at once. "
             "Stage 2 teaches them to <b>speak the LLM's language</b>. "
             f"The two giants stay <b style='color:{IMG}'>frozen</b> the whole time.")

    code_panel("the training losses (pseudo-code)", f'''\
# Stage 1 (no LLM): image features from the frozen ViT, plus the caption text
q = qformer(queries, image_feats)               # [B, {A['n_query']}, {A['q_hidden']}]
loss_itc = contrastive(q, text_cls)             # unimodal mask: queries and text apart
loss_itm = bce(match_head(qformer_joint(q, text)), is_match)   # bidirectional mask
loss_itg = lm_loss(qformer_causal(q, text), text)              # causal text, queries visible
loss = loss_itc + loss_itm + loss_itg

# Stage 2: plug in the frozen LLM
soft_prompt = projection(qformer(queries, image_feats))        # [B, {A['n_query']}, {A['llm_hidden']}]
logits = llm(inputs_embeds=cat([soft_prompt, embed(caption)]))
loss = cross_entropy(logits, caption)           # only Q-Former + projection get updated''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 · CAPTIONING
# ═════════════════════════════════════════════════════════════════════════════
def llm_input_html(prompt_parts, answer_words, answer_label):
    pr = "".join(prompt_parts) if prompt_parts else f'<span class="tk pr" style="font-style:italic">no text prompt</span>'
    ans = ("".join(f'<span class="tk ans">{esc(w)}</span>' for w in answer_words[:14]) +
           ("<span style='color:#64748B;font-weight:700'> ···</span>" if len(answer_words) > 14 else "")) \
        if answer_words else f'<span class="tk pr">?</span>'
    return f"""
    <div class="panel">
      <div class="ptitle">What the frozen LLM reads, and what it writes</div>
      <div class="pipe" style="align-items:stretch">
        <div class="node n-space" style="display:flex;flex-direction:column;justify-content:center">
          <div>{vtokens(8, A['n_query'], big=True)}</div><small>{A['n_query']} visual tokens from the Q-Former</small></div>
        <div class="arrow" style="align-self:center">+</div>
        <div class="node n-plain" style="display:flex;flex-direction:column;justify-content:center;max-width:340px">
          <div class="seq">{pr}</div><small>text prompt</small></div>
        <div class="arrow" style="align-self:center">→</div>
        <div class="node n-txt" style="display:flex;flex-direction:column;justify-content:center">OPT-2.7B<small>❄ frozen</small></div>
        <div class="arrow" style="align-self:center">→</div>
        <div class="node n-plain" style="display:flex;flex-direction:column;justify-content:center;max-width:360px">
          <div class="seq">{ans}</div><small>{answer_label}</small></div>
      </div>
    </div>"""


def sec_caption():
    hero(4, "Image captioning", "No text prompt: the LLM simply describes what the visual tokens show")
    if need_images():
        return
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        lbl = pick_image("s4_img")
        html(f'<img class="thumb" src="{thumb_uri(POOL[lbl]["bytes"], 260)}" style="width:100%;max-width:260px"/>')
    with c2:
        max_tok = st.slider("Max new tokens", 10, 80, 30, 5, key="s4_max")
        sample = st.toggle("Sampling (more variety)", value=True, key="s4_sample")
    with c3:
        temp = st.slider("Temperature", 0.3, 1.5, 1.0, 0.1, key="s4_temp", disabled=not sample)
        n = st.slider("How many captions", 1, 5, 3, key="s4_n", disabled=not sample)

    n = n if sample else 1
    sig = ("cap", lbl, max_tok, sample, temp if sample else None, n)
    if run_button("Generate captions", "s4_run"):
        with st.spinner("Image → ViT → Q-Former → OPT ..."):
            run_once("s4_res", sig, lambda: blip_generate(POOL[lbl]["bytes"], None, max_tok, sample, temp, n))
    caps = result("s4_res", sig)

    html(llm_input_html(None, caps[0].split() if caps else [], "generated caption"))
    if caps:
        html("".join(f'<div class="out"><small>Caption {k + 1}</small>{esc(c)}</div>' for k, c in enumerate(caps)))
    else:
        st.caption("Press the button to run BLIP-2. Results appear here.")

    key_idea("With no prompt, the LLM continues from the visual tokens alone, so it writes a caption. "
             "<b>Sampling</b> picks each next word at random from the model's probabilities. Higher "
             "<b>temperature</b> flattens those probabilities: more variety, more mistakes.")

    code_panel("captioning", f'''\
inputs = processor(images=image, return_tensors="pt").to(device, dtype)
out = model.generate(
    **inputs,
    max_new_tokens={max_tok},
    do_sample={sample},{f"""
    temperature={temp}, top_p=0.9,
    num_return_sequences={n},""" if sample else ""}
)
captions = processor.batch_decode(out, skip_special_tokens=True)''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 · QUESTION ANSWERING
# ═════════════════════════════════════════════════════════════════════════════
QUICK_QS = ["What is in this image?", "How many animals are there?", "What color is it?",
            "Where was this photo taken?", "What is happening?"]
FORMATS = {"Question: … Answer:": "Question: {q} Answer:", "Raw question": "{q}"}


def _set_q(q):
    st.session_state["s5_q"] = q


def sec_vqa():
    hero(5, "Visual question answering", "Same visual tokens, now followed by your question")
    if need_images():
        return
    c1, c2 = st.columns([1, 2.2])
    with c1:
        lbl = pick_image("s5_img")
        html(f'<img class="thumb" src="{thumb_uri(POOL[lbl]["bytes"], 240)}" style="width:100%;max-width:240px"/>')
    with c2:
        if "s5_q" not in st.session_state:
            st.session_state["s5_q"] = QUICK_QS[0]
        q = st.text_input("Your question", key="s5_q")
        qc = st.columns(len(QUICK_QS))
        for k, qq in enumerate(QUICK_QS):
            with qc[k]:
                st.button(qq, key=f"s5_quick_{k}", on_click=_set_q, args=(qq,))
        fmt = seg("Prompt format", list(FORMATS), key="s5_fmt")
        max_tok = st.slider("Max answer length (tokens)", 5, 60, 20, 5, key="s5_max")

    prompt = FORMATS[fmt].replace("{q}", q.strip())
    sig = ("vqa", lbl, prompt, max_tok)
    if run_button("Answer", "s5_run"):
        with st.spinner("Thinking..."):
            run_once("s5_res", sig, lambda: blip_generate(POOL[lbl]["bytes"], prompt, max_tok)[0])
    ans = result("s5_res", sig)

    parts = []
    for w in prompt.split():
        cls = "fmt" if w in ("Question:", "Answer:") else "pr"
        parts.append(f'<span class="tk {cls}">{esc(w)}</span>')
    html(llm_input_html(parts, ans.split() if ans else [], "answer"))
    if ans:
        html(f'<div class="out"><small>Q: {esc(q)}</small>{esc(ans)}</div>')
    else:
        st.caption("Press the button to run BLIP-2.")

    with st.expander("🧪  Experiment: ask several questions about the same image"):
        qs_txt = st.text_area("One question per line",
                              value="What is the main subject?\nWhat colors stand out?\n"
                                    "Describe the background.\nGive a one-word answer: what is this?",
                              height=120, key="s5_multi")
        qs = [x.strip() for x in qs_txt.splitlines() if x.strip()]
        sig2 = ("multi", lbl, tuple(qs), fmt)
        if run_button("Answer all", "s5_run_all"):
            with st.spinner(f"Answering {len(qs)} questions..."):
                run_once("s5_multi_res", sig2,
                         lambda: [blip_generate(POOL[lbl]["bytes"], FORMATS[fmt].replace("{q}", x), 30)[0] for x in qs])
        answers = result("s5_multi_res", sig2)
        if answers:
            m = max(len(a.split()) for a in answers) or 1
            rows = "".join(
                f'<tr><td style="padding:8px 10px;font-weight:600;color:{NAVY}">{esc(x)}</td>'
                f'<td style="padding:8px 10px">{esc(a)}</td>'
                f'<td style="padding:8px 10px;width:22%">{bars_html([f"{len(a.split())} words"], [len(a.split())], [TXT_BG], lambda v: "", m)}</td></tr>'
                for x, a in zip(qs, answers))
            html(f'<table class="cmp"><tr><th>question</th><th>answer</th><th>length</th></tr>{rows}</table>')

    key_idea("The frozen OPT was <b>never trained on question answering</b>. The "
             f"<b style='color:#B45309'>\"Question: … Answer:\"</b> format primes it to reply briefly. "
             "Switch to the raw question and compare: prompt wording changes what you get.")

    code_panel("visual question answering", f'''\
prompt = "{esc(prompt)}"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)
out = model.generate(**inputs, max_new_tokens={max_tok})
answer = processor.decode(out[0], skip_special_tokens=True)
answer = answer.split("Answer:")[-1].strip()   # some versions echo the prompt''')


def bars_html(names, values, colors, fmt, scale_max=None):
    m = scale_max or max(max(values), 1e-6)
    out = ""
    for nm, v, c in zip(names, values, colors):
        w = max(v, 0) / m * 100
        out += (f'<div class="bar"><div class="track"><div class="fill" style="width:{w:.0f}%;background:{c}"></div>'
                f'<span class="lab">{esc(nm)}</span></div><div class="v">{fmt(v)}</div></div>')
    return out


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 · BLIP-2 VS CLIP
# ═════════════════════════════════════════════════════════════════════════════
def sec_vs_clip():
    hero(6, "BLIP-2 vs CLIP", "Same image, two very different kinds of output")
    if need_images():
        return
    c1, c2, c3 = st.columns([1, 1.2, 1.4])
    with c1:
        lbl = pick_image("s6_img")
        html(f'<img class="thumb" src="{thumb_uri(POOL[lbl]["bytes"], 220)}" style="width:100%;max-width:220px"/>')
    with c2:
        labels_txt = st.text_area("Labels for CLIP (one per line)", value="\n".join(LABELS), height=170, key="s6_labels")
    with c3:
        question = st.text_input("Question for BLIP-2", value="What is in this image?", key="s6_q")
        st.caption("CLIP scores each label with the prompt \"a photo of a {label}.\"")

    labels = list(dict.fromkeys(x.strip() for x in labels_txt.splitlines() if x.strip()))
    sig = ("vs", lbl, tuple(labels), question)
    if run_button("Run both models", "s6_run"):
        def both():
            s, p = clip_scores(POOL[lbl]["bytes"], [f"a photo of {article(x)} {x}." for x in labels])
            cap = blip_generate(POOL[lbl]["bytes"], None, 30)[0]
            ans = blip_generate(POOL[lbl]["bytes"], f"Question: {question} Answer:", 20)[0]
            return s, p, cap, ans
        with st.spinner("Running CLIP and BLIP-2..."):
            run_once("s6_res", sig, both)
    res = result("s6_res", sig)

    left, right = st.columns(2)
    with left:
        if res:
            s, p, _, _ = res
            order = np.argsort(-p)[:6]
            bars = bars_html([labels[i] for i in order], [p[i] for i in order],
                             [POS_BG if k == 0 else "#E2E8F0" for k in range(len(order))],
                             lambda v: f"{v * 100:.1f}%", 1.0)
        else:
            bars = f'<div class="slabel">Press ▶ to see CLIP\'s ranking.</div>'
        html(f"""<div class="panel" style="border-top:6px solid {IMG}">
          <div class="ptitle"><span class="tag" style="background:{IMG_BG};color:{IMG}">CLIP</span>chooses from a list</div>
          <div class="pipe" style="margin-bottom:10px">
            <div class="node n-img">image encoder</div><div class="arrow">+</div>
            <div class="node n-txt">text encoder</div><div class="arrow">→</div>
            <div class="node n-space">cosine scores</div></div>
          {bars}</div>""")
    with right:
        if res:
            _, _, cap, ans = res
            outs = (f'<div class="out"><small>Caption (no prompt)</small>{esc(cap)}</div>'
                    f'<div class="out"><small>Q: {esc(question)}</small>{esc(ans)}</div>')
        else:
            outs = f'<div class="slabel">Press ▶ to see what BLIP-2 writes.</div>'
        html(f"""<div class="panel" style="border-top:6px solid {TXT}">
          <div class="ptitle"><span class="tag" style="background:{TXT_BG};color:#0B7A55">BLIP-2</span>writes new words</div>
          <div class="pipe" style="margin-bottom:10px">
            <div class="node n-img">ViT-g ❄</div><div class="arrow">→</div>
            <div class="node" style="background:{PROJ_BG};border:2px solid {PROJ};color:#B45309">Q-Former 🔥</div>
            <div class="arrow">→</div><div class="node n-txt">OPT ❄</div><div class="arrow">→</div>
            <div class="node n-plain">text</div></div>
          {outs}</div>""")

    rows = [
        ("Output", "Similarity scores between embeddings", "Generated text"),
        ("Main tasks", "Retrieval, ranking, zero-shot classification", "Captioning, question answering"),
        ("Image encoder", "ViT-B/32, trained from scratch with the text encoder", "ViT-g/14 from EVA-CLIP, frozen"),
        ("Language side", "Text encoder, trained jointly", "Frozen OPT-2.7B LLM"),
        ("Bridge", "None: two projections into one shared space", "Q-Former with 32 learned queries"),
        ("Training signal", "Contrastive (InfoNCE) only", "Stage 1: ITC + ITM + ITG · Stage 2: next-word prediction"),
        ("Answers limited to", "The labels you supply", "Anything the LLM can write"),
    ]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    html(f'<table class="cmp"><tr><th></th><th style="color:{IMG}">CLIP</th>'
         f'<th style="color:#0B7A55">BLIP-2</th></tr>{body}</table>')

    key_idea(f"<b style='color:{IMG}'>CLIP chooses</b>: it can only rank the options you give it. "
             f"<b style='color:{TXT}'>BLIP-2 writes</b>: it generates new words. In fact BLIP-2's stage 1 "
             "contains a CLIP-style contrastive loss, so it builds on the same idea.")

    code_panel("running both", '''\
# CLIP: score fixed labels
probs = (clip.logit_scale.exp() * img_emb @ text_embs.T).softmax(-1)

# BLIP-2: generate free text
caption = processor.decode(model.generate(**processor(images=image, return_tensors="pt"))[0],
                           skip_special_tokens=True)''')


# ═════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
SECTIONS = {
    "1 · Big picture": sec_big_picture,
    "2 · Inside the Q-Former": sec_qformer,
    "3 · Two-stage training": sec_training,
    "4 · Captioning": sec_caption,
    "5 · Question answering": sec_vqa,
    "6 · BLIP-2 vs CLIP": sec_vs_clip,
}
NAMES = list(SECTIONS)

if "sec" not in st.session_state:
    st.session_state.sec = NAMES[0]
if "last_sec" not in st.session_state:
    st.session_state.last_sec = NAMES[0]


def _keep_sec():
    if st.session_state.sec is None:          # segmented control was clicked off
        st.session_state.sec = st.session_state.last_sec
    st.session_state.last_sec = st.session_state.sec


def _go(delta):
    i = NAMES.index(st.session_state.sec) + delta
    st.session_state.sec = NAMES[max(0, min(len(NAMES) - 1, i))]
    st.session_state.last_sec = st.session_state.sec


try:
    nav_box = st.container(key="topnav")
except TypeError:                              # Streamlit < 1.39 has no container keys
    nav_box = st.container()
with nav_box:
    if hasattr(st, "segmented_control"):
        st.segmented_control("Section", NAMES, key="sec", on_change=_keep_sec, label_visibility="collapsed")
    else:
        st.radio("Section", NAMES, key="sec", on_change=_keep_sec, horizontal=True, label_visibility="collapsed")

current = st.session_state.sec or st.session_state.last_sec
SECTIONS[current]()

st.markdown("")
idx = NAMES.index(current)
b1, _, b2 = st.columns([1, 4, 1])
with b1:
    if idx > 0:
        st.button(f"← {NAMES[idx - 1][4:]}", on_click=_go, args=(-1,), key="prev")
with b2:
    if idx < len(NAMES) - 1:
        st.button(f"{NAMES[idx + 1][4:]} →", on_click=_go, args=(1,), key="next")

