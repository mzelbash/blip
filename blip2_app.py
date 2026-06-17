"""
BLIP-2 Interactive Implementation Walkthrough
SEAS 8525: Computer Vision and Generative AI
Dr. Elbasheer -- Week 6

Setup:
    pip install streamlit torch torchvision transformers
    pip install matplotlib seaborn pillow numpy accelerate

Run:
    streamlit run blip2_app.py

Notes:
    Model: Salesforce/blip2-opt-2.7b (~3.7GB download on first run)
    GPU strongly recommended -- CPU inference takes 20-40 seconds per call.
    On first launch the model downloads automatically from HuggingFace.

On Google Colab (GPU runtime recommended):
    !pip install streamlit pyngrok transformers accelerate torch torchvision
    !streamlit run blip2_app.py &
    from pyngrok import ngrok
    print(ngrok.connect(8501))
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BLIP-2 Walkthrough | SEAS 8525",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SIDEBAR WIDTH STATE ───────────────────────────────────────────────────────
if "sidebar_width" not in st.session_state:
    st.session_state.sidebar_width = 280

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{
        min-width: {st.session_state.sidebar_width}px !important;
        max-width: {st.session_state.sidebar_width}px !important;
        width:     {st.session_state.sidebar_width}px !important;
        background: #1A2F4A;
        border-right: 2px solid #0D2B4E;
        transition: width 0.15s ease;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Base */
    [data-testid="stAppViewContainer"] { background: #F7F9FC; color: #1A2332; }
    [data-testid="stSidebar"] {
        background: #1A2F4A;
        border-right: 2px solid #0D2B4E;
        transition: width 0.15s ease;
    }
    [data-testid="stSidebar"] * { color: #E8EFF7 !important; }
    [data-testid="stSidebar"] .stCodeBlock,
    [data-testid="stSidebar"] pre,
    [data-testid="stSidebar"] code {
        background: #0A1628 !important;
        color: #80CBC4 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; }

    /* Main text */
    h1 { color: #0D47A1 !important; font-family: Georgia, serif; }
    h2 { color: #1565C0 !important; font-family: Georgia, serif; }
    h3 { color: #1976D2 !important; }
    p, li { color: #1A2332; font-size: 15px; line-height: 1.7; }

    /* Info boxes */
    .concept-box {
        background: #E3F2FD;
        border-left: 5px solid #0097A7;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #0D2B4E;
        font-size: 14px;
        line-height: 1.7;
    }
    .highlight-box {
        background: #EDE7F6;
        border-left: 5px solid #5E35B1;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #1A0050;
        font-size: 14px;
        line-height: 1.7;
    }
    .warning-box {
        background: #FFF8E1;
        border-left: 5px solid #F9A825;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #3E2723;
        font-size: 13px;
    }
    .success-box {
        background: #E8F5E9;
        border-left: 5px solid #2E7D32;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #1B3A1C;
        font-size: 13px;
    }
    .metric-card {
        background: #FFFFFF;
        border: 2px solid #BBDEFB;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 6px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .metric-val {
        font-size: 26px;
        font-weight: bold;
        color: #0D47A1;
    }
    .metric-label {
        font-size: 12px;
        color: #607D8B;
        margin-top: 4px;
    }
    .section-tag {
        display: inline-block;
        background: #0097A7;
        color: white;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    /* Code blocks: white background, readable */
    .stCodeBlock, pre, code {
        background: #F1F5F9 !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px;
    }
    .stButton > button {
        background: #1565C0;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        padding: 8px 20px;
    }
    .stButton > button:hover { background: #1976D2; }
    hr { border-color: #BBDEFB; }
    [data-testid="stFileUploader"] { background: #FFFFFF; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_blip2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, processor, device


def generate_caption(model, processor, image, device, max_new_tokens=60):
    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = processor(images=image, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def answer_question(model, processor, image, question, device, max_new_tokens=80):
    dtype = torch.float16 if device == "cuda" else torch.float32
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(device, dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    full = processor.decode(out[0], skip_special_tokens=True).strip()
    # Strip the prompt prefix if echoed back
    if "Answer:" in full:
        full = full.split("Answer:")[-1].strip()
    return full


def make_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7F9FC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
    ax.tick_params(colors="#334155", labelsize=10)
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")
    ax.title.set_color("#0D47A1")
    return fig, ax


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.session_state.sidebar_width = st.slider(
        "Panel width", min_value=200, max_value=440,
        value=st.session_state.sidebar_width, step=10,
        help="Drag to resize the sidebar"
    )
    st.markdown("## 🧠 BLIP-2 Walkthrough")
    st.markdown("**SEAS 8525** · Computer Vision and Generative AI")
    st.markdown("*Dr. Elbasheer · Week 6*")
    st.divider()

    section = st.radio(
        "Jump to section",
        options=[
            "1 · What is BLIP-2?",
            "2 · The QFormer Bridge",
            "3 · Image Captioning",
            "4 · Visual Question Answering",
            "5 · Prompt Engineering",
            "6 · BLIP-2 vs CLIP",
        ]
    )
    st.divider()

    st.markdown("#### Quick Setup")
    st.code(
        "pip install streamlit torch torchvision\n"
        "pip install transformers accelerate\n"
        "pip install matplotlib pillow numpy\n\n"
        "streamlit run blip2_app.py",
        language="bash"
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda":
        st.success(f"GPU ready: {torch.cuda.get_device_name(0)}")
    else:
        st.warning(
            "Running on CPU. Each inference call takes 20-40 seconds. "
            "GPU is strongly recommended for live demos."
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: WHAT IS BLIP-2?
# ═════════════════════════════════════════════════════════════════════════════
if section.startswith("1"):
    st.markdown('<span class="section-tag">SECTION 1</span>', unsafe_allow_html=True)
    st.title("BLIP-2: Bootstrapping Language-Image Pre-training")
    st.caption("Li et al., Salesforce Research 2023 · Model: Salesforce/blip2-opt-2.7b")

    st.markdown("""<div class="concept-box">
    <strong>Core idea:</strong> Connect a frozen image encoder and a frozen large language model
    using a lightweight trainable bridge called the QFormer. Neither backbone is updated during
    training; only the 188M-parameter QFormer learns. This makes BLIP-2 far cheaper to train
    than end-to-end multimodal models while achieving strong performance on captioning and VQA.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-val">188M</div><div class="metric-label">QFormer parameters (trained)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-val">32</div><div class="metric-label">Learnable query tokens</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-val">2</div><div class="metric-label">Training stages</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-val">❄️ x2</div><div class="metric-label">Frozen backbones</div></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("The Three-Part Architecture")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="highlight-box">
        <strong>❄️ Frozen Image Encoder</strong><br><br>
        ViT-G/14 from EVA-CLIP.<br>
        1.8 billion parameters.<br>
        Never updated during BLIP-2 training.<br>
        Outputs a set of visual patch features.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="concept-box">
        <strong>🔥 QFormer (Trainable)</strong><br><br>
        188M parameters, the only part that trains.<br>
        32 learnable query tokens attend to image features via cross-attention.<br>
        Outputs 32 fixed-length vectors regardless of image size.<br>
        Shared self-attention layers for text and queries.
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""<div class="highlight-box">
        <strong>❄️ Frozen LLM</strong><br><br>
        OPT-2.7B in this app (also available with FlanT5).<br>
        2.7 billion parameters.<br>
        Never updated during BLIP-2 training.<br>
        Receives QFormer output as soft prompt tokens.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Two-Stage Training Strategy")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Stage 1: Vision-Language Representation Learning**")
        st.markdown("""<div class="concept-box">
        No LLM involved. Just the frozen image encoder and QFormer.<br><br>
        Three simultaneous objectives:<br>
        1. Image-Text Contrastive (like CLIP, aligning matching pairs)<br>
        2. Image-Text Matching (binary: does this pair match?)<br>
        3. Image-Grounded Text Generation (generate caption from visual features)
        </div>""", unsafe_allow_html=True)

        st.code(
            "# Stage 1 -- QFormer learns to extract\n"
            "# language-relevant visual features\n\n"
            "# Input: image patches from frozen ViT\n"
            "# Queries: 32 learnable tokens\n"
            "# Output: 32 visual feature vectors\n\n"
            "# Three losses trained jointly:\n"
            "# L = L_itc + L_itm + L_itg",
            language="python"
        )

    with col_b:
        st.markdown("**Stage 2: Vision-to-Language Generative Learning**")
        st.markdown("""<div class="concept-box">
        Now the frozen LLM is connected.<br><br>
        QFormer output (32 vectors) passes through a linear projection layer
        and is prepended to the LLM input as soft visual prompt tokens.<br><br>
        The LLM sees these as additional context before the text.<br>
        Only QFormer and the linear projection are updated.
        </div>""", unsafe_allow_html=True)

        st.code(
            "# Stage 2 -- Connect QFormer to frozen LLM\n\n"
            "# 1. Encode image with frozen ViT\n"
            "visual_features = vit(image)       # [N_patches, 1408]\n\n"
            "# 2. QFormer extracts key info\n"
            "query_output = qformer(visual_features)  # [32, 768]\n\n"
            "# 3. Linear projection to LLM space\n"
            "soft_prompts = linear(query_output)  # [32, 2560]\n\n"
            "# 4. Prepend to LLM input\n"
            "output = llm(soft_prompts, text_tokens)",
            language="python"
        )

    st.divider()
    st.subheader("Loading BLIP-2 from HuggingFace")

    st.code(
        "from transformers import Blip2Processor, Blip2ForConditionalGeneration\n"
        "import torch\n\n"
        "processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')\n"
        "model = Blip2ForConditionalGeneration.from_pretrained(\n"
        "    'Salesforce/blip2-opt-2.7b',\n"
        "    torch_dtype=torch.float16,  # use float32 on CPU\n"
        "    device_map='auto'           # auto GPU placement\n"
        ")\n"
        "model.eval()\n\n"
        "# Model sizes:\n"
        "# Vision encoder (ViT-G/14):  ~1.8B params -- FROZEN\n"
        "# QFormer:                    ~188M params -- TRAINABLE\n"
        "# Language model (OPT-2.7B):  ~2.7B params -- FROZEN",
        language="python"
    )

    st.markdown("""<div class="warning-box">
    <strong>First run note:</strong> The model is approximately 3.7GB and downloads automatically
    from HuggingFace on first launch. Subsequent runs load from the local cache.
    GPU with at least 8GB VRAM is recommended for comfortable inference speed.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: THE QFORMER BRIDGE
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("2"):
    st.markdown('<span class="section-tag">SECTION 2</span>', unsafe_allow_html=True)
    st.title("The QFormer Bridge")
    st.caption("Inspecting the learnable bridge between the frozen vision encoder and the frozen LLM")

    st.markdown("""<div class="concept-box">
    The QFormer is the only part of BLIP-2 that is trained. It contains 32 learnable query tokens
    that attend to image patch features via cross-attention and extract the most language-relevant
    visual information. The output is always exactly 32 vectors, a fixed-length summary of
    the image that the LLM can process efficiently.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading BLIP-2, this may take a minute on first run..."):
        model, processor, device = load_blip2()
    st.success("BLIP-2 loaded: Salesforce/blip2-opt-2.7b")

    st.divider()
    st.subheader("Parameter Count: The Asymmetry That Makes BLIP-2 Efficient")

    components = {
        "Vision Encoder (ViT-G/14): FROZEN": sum(
            p.numel() for p in model.vision_model.parameters()
        ),
        "QFormer: TRAINABLE": sum(
            p.numel() for p in model.qformer.parameters()
        ),
        "Language Model (OPT-2.7B): FROZEN": sum(
            p.numel() for p in model.language_model.parameters()
        ),
    }

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    colors = ["#1565C0", "#FFB300", "#2E7D32"]
    labels = ["Frozen", "Trainable", "Frozen"]

    for i, (name, count) in enumerate(components.items()):
        with cols[i]:
            color = colors[i]
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val" style="color:{color}">{count/1e9:.2f}B</div>'
                f'<div class="metric-label">{name}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    total = sum(components.values())
    trainable = components["QFormer: TRAINABLE"]
    pct = trainable / total * 100

    st.markdown(f"""<div class="success-box">
    <strong>Only {pct:.1f}% of total parameters are trained.</strong>
    The QFormer ({trainable/1e6:.0f}M params) does all the learning while
    {(total-trainable)/1e9:.1f}B parameters in the two frozen backbones stay fixed.
    This is what makes BLIP-2 trainable on a fraction of the compute of end-to-end models.
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("QFormer Architecture Details")

    if st.button("Inspect QFormer config"):
        cfg = model.config.qformer_config
        st.json({
            "num_hidden_layers":      cfg.num_hidden_layers,
            "num_attention_heads":    cfg.num_attention_heads,
            "hidden_size":            cfg.hidden_size,
            "intermediate_size":      cfg.intermediate_size,
            "num_query_tokens":       model.config.num_query_tokens,
            "cross_attention_freq":   cfg.cross_attention_frequency,
            "encoder_hidden_size":    cfg.encoder_hidden_size,
        })

    st.divider()
    st.subheader("The 32 Learnable Query Tokens")

    st.markdown("""<div class="concept-box">
    The query tokens are randomly initialized embedding vectors that learn during training
    to ask the right questions of the image. They are not image patches; they are
    trainable parameters that selectively pull out the visual information most useful
    for language tasks via cross-attention over the ViT patch features.
    </div>""", unsafe_allow_html=True)

    st.code(
        "# Inspect the learnable query tokens\n"
        "query_tokens = model.query_tokens\n"
        "print(query_tokens.shape)\n"
        "# torch.Size([1, 32, 768])\n"
        "# 1 = batch dim, 32 = number of queries, 768 = hidden size\n\n"
        "# These 32 x 768 = 24,576 values are the only parameters\n"
        "# that directly interface with the image encoder output.\n"
        "# Everything the LLM knows about the image comes through them.",
        language="python"
    )

    if st.button("Show query token statistics"):
        qt = model.query_tokens.detach().cpu().float().numpy()
        st.markdown(f"""<div class="highlight-box">
        <strong>Query token tensor shape:</strong> {qt.shape}<br>
        <strong>Mean value:</strong> {qt.mean():.4f}<br>
        <strong>Std deviation:</strong> {qt.std():.4f}<br>
        <strong>Min:</strong> {qt.min():.4f} &nbsp;&nbsp; <strong>Max:</strong> {qt.max():.4f}<br><br>
        These values were randomly initialized and learned entirely during BLIP-2 training.
        </div>""", unsafe_allow_html=True)

        fig, ax = make_fig(9, 3)
        vals = qt[0].flatten()
        ax.hist(vals, bins=80, color="#1565C0", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Token value")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of learned query token values (32 tokens x 768 dims = 24,576 values)")
        ax.axvline(vals.mean(), color="#FFB300", linewidth=2, linestyle="--", label=f"Mean: {vals.mean():.3f}")
        ax.legend(labelcolor="#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Information Flow Through the QFormer")

    st.code(
        "# Full forward pass -- step by step\n\n"
        "# Step 1: encode image with frozen ViT\n"
        "pixel_values = processor(images=image, return_tensors='pt').pixel_values\n"
        "image_embeds = model.vision_model(pixel_values).last_hidden_state\n"
        "# shape: [1, 257, 1408]  -- 256 patches + 1 CLS token, 1408-dim features\n\n"
        "# Step 2: QFormer cross-attends to image features\n"
        "image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)\n"
        "query_tokens = model.query_tokens.expand(1, -1, -1)\n"
        "query_output = model.qformer(\n"
        "    query_embeds=query_tokens,\n"
        "    encoder_hidden_states=image_embeds,\n"
        "    encoder_attention_mask=image_atts,\n"
        ")\n"
        "# shape: [1, 32, 768]  -- 32 queries distilled image info\n\n"
        "# Step 3: project to LLM input space\n"
        "language_model_inputs = model.language_projection(query_output.last_hidden_state)\n"
        "# shape: [1, 32, 2560]  -- ready for OPT-2.7B\n\n"
        "# Step 4: frozen LLM generates text\n"
        "outputs = model.language_model.generate(inputs_embeds=language_model_inputs)",
        language="python"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: IMAGE CAPTIONING
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("3"):
    st.markdown('<span class="section-tag">SECTION 3</span>', unsafe_allow_html=True)
    st.title("Image Captioning")
    st.caption("Upload any image and let BLIP-2 describe it using the full frozen pipeline")

    st.markdown("""<div class="concept-box">
    Image captioning is BLIP-2's most direct capability. The frozen ViT encodes the image,
    the QFormer distills it into 32 visual tokens, a linear projection maps them to the LLM
    input space, and the frozen OPT-2.7B generates a natural language description.
    No text prompt is needed; the model generates freely from the visual tokens alone.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading BLIP-2..."):
        model, processor, device = load_blip2()

    st.divider()
    uploaded = st.file_uploader("Upload an image to caption", type=["png","jpg","jpeg","webp"])

    col_img, col_cfg = st.columns([1, 1])

    with col_img:
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

    with col_cfg:
        st.subheader("Generation settings")
        max_tokens = st.slider("Max new tokens", 20, 150, 60, 10)
        n_captions = st.slider("Number of captions to generate", 1, 5, 3,
                               help="Generate multiple captions with sampling to see variation")
        do_sample  = st.checkbox("Use sampling (adds diversity)", value=True)
        temperature = st.slider("Temperature", 0.5, 1.5, 1.0, 0.1,
                                disabled=not do_sample)

        st.markdown("""<div class="warning-box">
        <strong>CPU note:</strong> Each caption takes 20-40 seconds on CPU.
        GPU inference is under 2 seconds.
        </div>""", unsafe_allow_html=True)

    if uploaded and st.button("Generate Captions"):
        img = Image.open(uploaded).convert("RGB")
        dtype = torch.float16 if device == "cuda" else torch.float32
        inputs = processor(images=img, return_tensors="pt").to(device, dtype)

        captions = []
        with st.spinner(f"Generating {n_captions} caption(s)..."):
            for i in range(n_captions):
                with torch.no_grad():
                    if do_sample:
                        out = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                        )
                    else:
                        out = model.generate(**inputs, max_new_tokens=max_tokens)
                cap = processor.decode(out[0], skip_special_tokens=True).strip()
                captions.append(cap)

        st.divider()
        st.subheader("Generated Captions")
        for i, cap in enumerate(captions):
            st.markdown(f"""<div class="success-box">
            <strong>Caption {i+1}:</strong> {cap}
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("What Happened Under the Hood")
        st.code(
            "# 1. Processor prepares the image\n"
            "inputs = processor(images=image, return_tensors='pt')\n"
            "# pixel_values shape: [1, 3, 224, 224]\n\n"
            "# 2. Full pipeline: ViT -> QFormer -> Linear -> OPT\n"
            "output_ids = model.generate(\n"
            "    **inputs,\n"
            f"    max_new_tokens={max_tokens},\n"
            f"    do_sample={do_sample},\n"
            + (f"    temperature={temperature},\n" if do_sample else "") +
            ")\n\n"
            "# 3. Decode token IDs back to text\n"
            "caption = processor.decode(output_ids[0], skip_special_tokens=True)",
            language="python"
        )

        st.markdown("""<div class="concept-box">
        <strong>Why do captions vary?</strong> When sampling is on, the LLM draws from a
        probability distribution over next tokens at each step rather than always picking
        the most likely token. Temperature controls how spread out that distribution is.
        Higher temperature = more diverse but sometimes less accurate. Lower = more
        deterministic but potentially repetitive.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: VISUAL QUESTION ANSWERING
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("4"):
    st.markdown('<span class="section-tag">SECTION 4</span>', unsafe_allow_html=True)
    st.title("Visual Question Answering")
    st.caption("Ask any natural language question about an image")

    st.markdown("""<div class="concept-box">
    VQA is where BLIP-2's language model capabilities really show. You provide both an image
    and a text question. The QFormer extracts visual features, the linear projection maps them
    to the LLM input space, and the question is appended as text tokens. The frozen OPT
    generates a natural language answer conditioned on both.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading BLIP-2..."):
        model, processor, device = load_blip2()

    st.divider()
    col_l, col_r = st.columns([1, 1.1])

    with col_l:
        st.subheader("Upload Image")
        uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg","webp"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

    with col_r:
        st.subheader("Ask a Question")
        default_questions = [
            "What is in this image?",
            "What color is the main object?",
            "How many people are in the image?",
            "What is happening in this scene?",
            "What time of day does this appear to be?",
        ]
        question = st.text_input("Your question:", value="What is in this image?")

        st.markdown("**Or try one of these:**")
        for q in default_questions:
            if st.button(q, key=f"btn_{q}"):
                question = q

        max_tokens_vqa = st.slider("Max answer length (tokens)", 20, 120, 50, 10)

    if uploaded and question and st.button("Get Answer"):
        img = Image.open(uploaded).convert("RGB")

        with st.spinner("Thinking..."):
            answer = answer_question(model, processor, img, question, device, max_tokens_vqa)

        st.divider()
        st.markdown(f"""<div class="success-box">
        <strong>Question:</strong> {question}<br><br>
        <strong>Answer:</strong> {answer}
        </div>""", unsafe_allow_html=True)

        st.code(
            "# VQA forward pass\n"
            f'prompt = "Question: {question} Answer:"\n\n'
            "inputs = processor(\n"
            "    images=image,\n"
            "    text=prompt,\n"
            "    return_tensors='pt'\n"
            ")\n\n"
            "output_ids = model.generate(\n"
            "    **inputs,\n"
            f"    max_new_tokens={max_tokens_vqa}\n"
            ")\n"
            "answer = processor.decode(output_ids[0], skip_special_tokens=True)",
            language="python"
        )

        st.markdown("""<div class="concept-box">
        <strong>How it works:</strong> The prompt format "Question: ... Answer:" is a soft
        instruction that primes the frozen OPT to produce a concise answer rather than
        a free-form caption. The visual tokens from the QFormer are prepended to this
        text prompt so the LLM attends to both image and question simultaneously.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Multi-Question Comparison")
    st.markdown("Ask several questions about the same image at once to see how the model handles different reasoning types.")

    uploaded2 = st.file_uploader("Upload image for comparison", type=["png","jpg","jpeg","webp"], key="vqa2")

    default_multi = (
        "What is the main subject of this image?\n"
        "What colors are most prominent?\n"
        "Describe the background.\n"
        "What mood does this image convey?"
    )
    multi_qs = st.text_area("Questions (one per line):", value=default_multi, height=130)

    if uploaded2 and st.button("Answer All Questions"):
        img2 = Image.open(uploaded2).convert("RGB")
        questions = [q.strip() for q in multi_qs.strip().split("\n") if q.strip()]

        st.image(img2, width=300)
        st.divider()

        for q in questions:
            with st.spinner(f"Answering: {q}"):
                ans = answer_question(model, processor, img2, q, device, 60)
            st.markdown(f"""<div class="highlight-box">
            <strong>Q:</strong> {q}<br>
            <strong>A:</strong> {ans}
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: PROMPT ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("5"):
    st.markdown('<span class="section-tag">SECTION 5</span>', unsafe_allow_html=True)
    st.title("Prompt Engineering for BLIP-2")
    st.caption("How question phrasing changes the answer (same image, different prompts)")

    st.markdown("""<div class="concept-box">
    BLIP-2 uses a frozen OPT language model whose behavior is shaped by how the prompt
    is constructed. The same visual content can yield very different answers depending
    on how the question is phrased. This section lets you explore that sensitivity
    directly, the same principle as prompt engineering for text-only LLMs, but
    now applied across both vision and language.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading BLIP-2..."):
        model, processor, device = load_blip2()

    st.divider()
    uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=350)

    st.subheader("Compare Prompt Variants")
    st.markdown("Define a core topic and several ways of asking about it. BLIP-2 will answer each.")

    col_l, col_r = st.columns(2)
    with col_l:
        topic = st.text_input("Core topic:", value="the main object")
    with col_r:
        max_tok = st.slider("Max tokens per answer", 20, 100, 50, 10)

    default_prompts = (
        "What is {topic}?\n"
        "Describe {topic} in detail.\n"
        "What can you tell me about {topic}?\n"
        "Is there {topic} in this image?\n"
        "Give a one-word answer: what is {topic}?"
    )
    prompt_template = st.text_area(
        "Prompt variants (use {topic} as placeholder, one per line):",
        value=default_prompts, height=160
    )

    if uploaded and st.button("Run Prompt Comparison"):
        img = Image.open(uploaded).convert("RGB")
        raw_prompts = [
            p.strip().replace("{topic}", topic)
            for p in prompt_template.strip().split("\n") if p.strip()
        ]

        st.divider()
        st.subheader("Results")

        answers = []
        for prompt in raw_prompts:
            with st.spinner(f"Running: {prompt[:50]}..."):
                ans = answer_question(model, processor, img, prompt, device, max_tok)
            answers.append((prompt, ans))

        for prompt, ans in answers:
            st.markdown(f"""<div class="highlight-box">
            <strong>Prompt:</strong> {prompt}<br>
            <strong>Answer:</strong> {ans}
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("Answer Length Comparison")
        fig, ax = make_fig(8, 3.5)
        lengths = [len(ans.split()) for _, ans in answers]
        short_prompts = [p[:35] + "..." if len(p) > 35 else p for p, _ in answers]
        bars = ax.barh(range(len(answers)), lengths, color="#1565C0", height=0.5)
        ax.set_yticks(range(len(answers)))
        ax.set_yticklabels(short_prompts, color="#334155", fontsize=10)
        ax.set_xlabel("Answer length (words)")
        ax.set_title("How prompt phrasing affects answer verbosity")
        for i, v in enumerate(lengths):
            ax.text(v + 0.3, i, str(v), va="center", color="#334155", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""<div class="warning-box">
        <strong>Key observation:</strong> Open-ended prompts ("Describe...") produce longer,
        richer answers. Closed-ended prompts ("Is there...") produce shorter confirmations.
        Explicit constraints ("Give a one-word answer") are respected by the LLM but not
        always perfectly; the frozen backbone was not instruction-tuned for strict
        output formatting.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Captioning vs VQA Mode")
    st.markdown("""<div class="concept-box">
    BLIP-2 operates in two modes depending on whether a text prompt is provided:<br><br>
    <strong>No prompt (captioning mode):</strong> The LLM generates freely from visual tokens alone.
    Output tends to be a descriptive sentence starting with "a photo of..." or similar.<br><br>
    <strong>With prompt (VQA mode):</strong> The text prompt is appended after the visual tokens.
    The LLM generates text conditioned on both image and question.
    Use "Question: ... Answer:" format for best VQA results.
    </div>""", unsafe_allow_html=True)

    st.code(
        "# Captioning mode -- no text prompt\n"
        "inputs = processor(images=image, return_tensors='pt')\n"
        "output = model.generate(**inputs, max_new_tokens=50)\n\n"
        "# VQA mode -- with text prompt\n"
        "inputs = processor(\n"
        "    images=image,\n"
        '    text="Question: What is in this image? Answer:",\n'
        "    return_tensors='pt'\n"
        ")\n"
        "output = model.generate(**inputs, max_new_tokens=50)",
        language="python"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: BLIP-2 VS CLIP
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("6"):
    st.markdown('<span class="section-tag">SECTION 6</span>', unsafe_allow_html=True)
    st.title("BLIP-2 vs CLIP: Two Different Outputs from the Same Image")
    st.caption("Making the architectural difference concrete")

    st.markdown("""<div class="concept-box">
    CLIP and BLIP-2 both process images and text, but they produce fundamentally different
    outputs. CLIP produces embedding vectors for similarity and retrieval.
    BLIP-2 produces generated natural language for understanding and description.
    This section runs both on the same image so you can see the difference directly.
    </div>""", unsafe_allow_html=True)

    # Load BLIP-2
    with st.spinner("Loading BLIP-2..."):
        model, processor, device = load_blip2()

    # Load CLIP separately
    @st.cache_resource(show_spinner=True)
    def load_clip_for_comparison():
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        return clip_model, clip_proc

    with st.spinner("Loading CLIP..."):
        clip_model, clip_proc = load_clip_for_comparison()

    st.success("Both models loaded.")
    st.divider()

    uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])
    candidate_texts = st.text_area(
        "Candidate labels for CLIP (one per line):",
        value=(
            "a photo of a dog\n"
            "a photo of a cat\n"
            "a photo of a car\n"
            "a photo of food\n"
            "a photo of a person\n"
            "a photo of nature"
        ),
        height=160
    )
    question_for_blip = st.text_input(
        "Question for BLIP-2 VQA:",
        value="What is in this image?"
    )

    if uploaded and st.button("Run Both Models"):
        img = Image.open(uploaded).convert("RGB")

        col_l, col_r = st.columns(2)
        with col_l:
            st.image(img, use_container_width=True, caption="Input image")

        # Run CLIP
        candidates = [c.strip() for c in candidate_texts.strip().split("\n") if c.strip()]
        with st.spinner("Running CLIP..."):
            clip_inputs = clip_proc(
                images=img, text=candidates,
                return_tensors="pt", padding=True
            )
            with torch.no_grad():
                img_feat  = clip_model.get_image_features(pixel_values=clip_inputs["pixel_values"])
                txt_feats = clip_model.get_text_features(
                    input_ids=clip_inputs["input_ids"],
                    attention_mask=clip_inputs["attention_mask"]
                )
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            scores = (img_feat @ txt_feats.T).squeeze().cpu().numpy()

        # Run BLIP-2 captioning
        with st.spinner("Running BLIP-2 captioning..."):
            caption = generate_caption(model, processor, img, device)

        # Run BLIP-2 VQA
        with st.spinner("Running BLIP-2 VQA..."):
            vqa_answer = answer_question(
                model, processor, img, question_for_blip, device
            )

        st.divider()
        col_clip, col_blip = st.columns(2)

        with col_clip:
            st.subheader("CLIP Output")
            st.markdown("""<div class="highlight-box">
            <strong>What CLIP produces:</strong> Similarity scores.<br>
            CLIP cannot generate text. It ranks candidate labels by how
            closely each matches the image in the shared embedding space.
            </div>""", unsafe_allow_html=True)

            sorted_idx = np.argsort(scores)[::-1]
            fig, ax = make_fig(5, max(3, len(candidates) * 0.55))
            y_pos = list(range(len(candidates)))
            sorted_scores = scores[sorted_idx]
            sorted_labels = [candidates[i] for i in sorted_idx]
            bar_cols = (
                ["#1B5E20"] +
                ["#1565C0"] * min(2, len(candidates)-1) +
                ["#90A4AE"] * max(0, len(candidates)-3)
            )
            ax.barh(y_pos, sorted_scores[::-1], color=list(reversed(bar_cols)), height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_labels[::-1], color="#334155", fontsize=10)
            ax.set_xlabel("Cosine similarity")
            ax.set_title("CLIP: ranked similarity scores")
            for i, s in enumerate(sorted_scores[::-1]):
                ax.text(s + 0.002, i, f"{s:.3f}", va="center", color="#1A2332", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown(f"""<div class="success-box">
            <strong>Top match:</strong> {sorted_labels[0]}<br>
            <strong>Score:</strong> {sorted_scores[0]:.4f}<br><br>
            Output is a number between -1 and +1. No language generated.
            </div>""", unsafe_allow_html=True)

        with col_blip:
            st.subheader("BLIP-2 Output")
            st.markdown("""<div class="highlight-box">
            <strong>What BLIP-2 produces:</strong> Generated natural language.<br>
            BLIP-2 does not rank candidates; it generates free-form text
            using the frozen OPT language model conditioned on visual tokens.
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="success-box">
            <strong>Caption (no prompt):</strong><br>
            {caption}
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="success-box">
            <strong>VQA answer:</strong><br>
            Q: {question_for_blip}<br>
            A: {vqa_answer}
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("Architecture Comparison")

        rows = [
            ["", "CLIP", "BLIP-2"],
            ["Output type", "Embedding vector (512-dim)", "Generated text (natural language)"],
            ["Primary task", "Retrieval and similarity", "Captioning and VQA"],
            ["Image encoder", "ViT-B/32 (trained jointly)", "ViT-G/14 (frozen)"],
            ["Language side", "Text encoder (frozen after training)", "Frozen OPT-2.7B LLM"],
            ["Bridge", "None (shared projection space)", "QFormer (32 learnable queries)"],
            ["Training signal", "Contrastive (InfoNCE)", "Generation (cross-entropy)"],
            ["Zero-shot?", "Yes, any candidate labels", "Yes, no fine-tuning required"],
        ]

        st.table(
            {
                "": [r[0] for r in rows[1:]],
                "CLIP": [r[1] for r in rows[1:]],
                "BLIP-2": [r[2] for r in rows[1:]],
            }
        )

        st.markdown("""<div class="concept-box">
        <strong>When to use which:</strong><br>
        Use <strong>CLIP</strong> when your task is retrieval, ranking, filtering, or zero-shot
        classification, any time a similarity score is sufficient.<br><br>
        Use <strong>BLIP-2</strong> when your task requires understanding and generating language
        about an image: captioning, VQA, visual reasoning, or any task where a score
        is not enough and you need words.
        </div>""", unsafe_allow_html=True)
