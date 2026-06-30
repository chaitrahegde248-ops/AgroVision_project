import os
import io
import base64
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
from torchvision import models, transforms
import pandas as pd
import numpy as np
from torch import nn

st.set_page_config(
    page_title="AgroVision | Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0f1117 !important;
    color: #e2e8f0 !important;
}

#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container { max-width: 880px !important; padding: 0 2rem 5rem !important; }

/* Navbar */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 1.6rem; background: #1a1d27;
    border: 1px solid #2d3148; border-radius: 14px; margin-bottom: 2.5rem;
}
.nav-brand {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem;
    font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 9px;
}
.nav-dot {
    width: 8px; height: 8px; background: #22c55e;
    border-radius: 50%; display: inline-block; box-shadow: 0 0 6px #22c55e;
}
.nav-tag {
    font-size: 0.68rem; font-weight: 600; color: #cbd5e1;
    background: #252836; border: 1px solid #2d3148;
    padding: 4px 12px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.12em;
}

/* Hero */
.hero { text-align: center; padding: 2.5rem 1rem; }
.hero-eyebrow {
    display: inline-block; font-size: 0.72rem; font-weight: 600; color: #22c55e;
    text-transform: uppercase; letter-spacing: 0.15em;
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25);
    padding: 4px 14px; border-radius: 20px; margin-bottom: 1.1rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif; font-size: 2.9rem;
    font-weight: 700; color: #f8fafc; line-height: 1.1; margin-bottom: 1rem;
}
.hero h1 em {
    font-style: normal;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero p { font-size: 1rem; color: #94a3b8; max-width: 440px; margin: 0 auto; line-height: 1.7; }

/* Card */
.card {
    background: #1a1d27; border: 1px solid #2d3148;
    border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
}
.card-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1.2rem; padding-bottom: 1rem; border-bottom: 1px solid #252836;
}
.card-icon {
    width: 34px; height: 34px; background: #252836; border: 1px solid #2d3148;
    border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 0.95rem;
}
.card-title { font-size: 0.75rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.12em; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: #12141e !important; border-radius: 10px !important;
    padding: 4px !important; gap: 4px !important; border: 1px solid #2d3148 !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: #94a3b8 !important; font-weight: 600 !important;
    font-size: 0.88rem !important; border-radius: 8px !important; padding: 0.5rem 1.2rem !important;
    border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #22c55e !important; color: #000 !important;
}
[data-testid="stTabs"] [role="tab"] p,
[data-testid="stTabs"] [role="tab"] span { color: inherit !important; }

/* Uploader */
[data-testid="stFileUploader"] {
    background: #ffffff !important; border: 2px dashed #3a3f5c !important; border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #22c55e !important; }
[data-testid="stFileUploaderDropzone"] { background: #ffffff !important; }
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p { color: #000000 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: #555555 !important; font-size: 0.75rem !important; }
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button span { color: #16a34a !important; font-weight: 600 !important; }

/* Image */
[data-testid="stImage"] img {
    border-radius: 12px !important; border: 1px solid #2d3148 !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}

/* Progress */
div[data-testid="stProgress"] { height: auto !important; min-height: 0 !important; padding: 0 !important; }
div[data-testid="stProgress"] > div[role="progressbar"] {
    height: 8px !important; background: #252836 !important;
    border-radius: 99px !important; overflow: hidden !important;
}
div[data-testid="stProgress"] > div[role="progressbar"] > div {
    height: 8px !important; background: linear-gradient(90deg, #16a34a, #22c55e) !important;
    border-radius: 99px !important;
}

/* Alerts */
.stSuccess > div {
    background: rgba(34,197,94,0.08) !important; border: 1px solid rgba(34,197,94,0.25) !important;
    color: #86efac !important; border-radius: 10px !important;
}
.stInfo > div {
    background: rgba(56,189,248,0.08) !important; border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 10px !important; color: #bae6fd !important;
}

/* Result */
.result-wrap { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 1.4rem; }
.result-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3);
    border-radius: 50px; padding: 0.55rem 1.3rem; font-size: 1rem; font-weight: 700; color: #86efac;
}
.result-arrow { color: #475569; font-size: 1.1rem; }
.result-sub {
    background: #252836; border: 1px solid #2d3148; border-radius: 50px;
    padding: 0.45rem 1.1rem; font-size: 0.88rem; font-weight: 600; color: #94a3b8;
}

/* Stats */
.stats-row { display: flex; gap: 0.75rem; }
.stat-chip {
    flex: 1; background: #12141e; border: 1px solid #2d3148;
    border-radius: 12px; padding: 0.9rem 1rem; text-align: center;
}
.stat-label { font-size: 0.63rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; }
.stat-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin-top: 3px; }

/* Price */
.price-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
.price-box {
    background: #12141e; border: 1px solid #2d3148; border-top: 2px solid #22c55e;
    border-radius: 14px; padding: 1.6rem 1.2rem; text-align: center;
}
.price-month { font-size: 0.67rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.6rem; }
.price-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; color: #f8fafc; line-height: 1; }
.price-unit  { font-size: 0.68rem; color: #94a3b8; margin-top: 0.4rem; }
.trend-up    { color: #22c55e; font-size: 0.78rem; margin-top: 0.5rem; font-weight: 600; }
.trend-down  { color: #f87171; font-size: 0.78rem; margin-top: 0.5rem; font-weight: 600; }

/* Footer */
.footer { text-align: center; padding: 2rem 0 0.5rem; border-top: 1px solid #1e2130; margin-top: 2.5rem; }
.footer p { font-size: 0.75rem; color: #475569; }

/* Global override */
.stApp p, .stApp label, [data-testid="stMarkdownContainer"] p { color: #e2e8f0 !important; }
.hero-eyebrow { color: #22c55e !important; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model ────────────────────────────────────────────
crop_map   = {"rice": "Sona Masoori Rice", "maize": "Maize", "wheat": "Wheat"}
CROP_EMOJI = {"rice": "🌾", "maize": "🌽", "wheat": "🌿"}

@st.cache_resource(show_spinner=False)
def load_model():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 3)
    ckpt = torch.load(
        os.path.join(os.path.dirname(__file__), "resnet_crop_model_with_mapping.pth"),
        map_location="cpu"
    )
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt["idx_to_class"]

with st.spinner(""):
    model, idx_to_class = load_model()

# ── Navbar ──────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-brand">🌾 &nbsp;AgroVision <span class="nav-dot"></span></div>
    <div class="nav-tag">AI Crop Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Powered by ResNet50 + LSTM</div>
    <h1>Identify Crops &<br><em>Forecast Prices</em></h1>
    <p>Upload a field image for instant crop identification and two-month market price forecasting.</p>
</div>
""", unsafe_allow_html=True)

# ── Step 1 ──────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="card-header">
        <div class="card-icon">📁</div>
        <div class="card-title">Step 1 — Upload or Capture Crop Image</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_upload, tab_cam = st.tabs(["📁  Upload File", "📷  Use Webcam"])

image_file = None

with tab_upload:
    image_file = st.file_uploader(
        "Upload crop image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with tab_cam:
    cam_image = st.camera_input("Take a photo", label_visibility="collapsed")
    if cam_image is not None:
        image_file = cam_image

# ── Analysis ────────────────────────────────────────────────
if image_file:
    img = Image.open(image_file).convert("RGB")

    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">🖼️</div>
            <div class="card-title">Preview</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img, use_column_width=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        out   = model(tfm(img).unsqueeze(0))
        probs = torch.softmax(out, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
        predicted_crop = idx_to_class[pred_idx.item()]
        excel_crop     = crop_map[predicted_crop]
        confidence     = conf.item() * 100
        emoji          = CROP_EMOJI.get(predicted_crop, "🌱")

    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">🔬</div>
            <div class="card-title">Step 2 — Identification Result</div>
        </div>
        <div class="result-wrap">
            <span class="result-pill">{emoji} &nbsp;{predicted_crop.capitalize()}</span>
            <span class="result-arrow">→</span>
            <span class="result-sub">{excel_crop}</span>
        </div>
        <div class="stats-row">
            <div class="stat-chip"><div class="stat-label">Confidence</div><div class="stat-value">{confidence:.1f}%</div></div>
            <div class="stat-chip"><div class="stat-label">Model</div><div class="stat-value">ResNet50</div></div>
            <div class="stat-chip"><div class="stat-label">Classes</div><div class="stat-value">3</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df      = pd.read_excel(os.path.join(os.path.dirname(__file__), "price.xlsx.xlsx"))
    crop_df = df[df["Crop"] == excel_crop].sort_values("Date")

    if len(crop_df) < 4:
        st.warning("Not enough historical data for price forecasting.")
        st.stop()

    prices        = crop_df["Price (INR/quintal)"].values.astype(float)
    current_price = prices[-1]
    min_p, max_p  = prices.min(), prices.max()
    norm          = (prices - min_p) / (max_p - min_p)

    seq = 3
    X, Y = [], []
    for i in range(len(norm) - seq):
        X.append(norm[i:i+seq]); Y.append(norm[i+seq])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

    class PriceLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.1)
            self.fc   = nn.Linear(64, 1)
        def forward(self, x):
            h0 = torch.zeros(2, x.size(0), 64)
            c0 = torch.zeros(2, x.size(0), 64)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    lstm      = PriceLSTM()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">⚙️</div>
            <div class="card-title">Step 3 — Training Price Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bar    = st.progress(0)
    status = st.empty()
    epochs = 300
    for epoch in range(epochs):
        lstm.train()
        loss = criterion(lstm(X), Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        pct = int((epoch + 1) / epochs * 100)
        if epoch % 15 == 0 or epoch == epochs - 1:
            bar.progress(pct)
            status.caption(f"Training... {pct}%  |  loss: {loss.item():.5f}")

    lstm.eval()
    last_seq  = X[-1].unsqueeze(0)
    raw_preds = []
    for _ in range(2):
        p = lstm(last_seq).item()
        raw_preds.append(p)
        ns = np.append(last_seq.squeeze(0).detach().numpy()[1:], [[p]], axis=0)
        last_seq = torch.tensor(ns, dtype=torch.float32).unsqueeze(0)

    forecast = [round(p * (max_p - min_p) + min_p, 2) for p in raw_preds]
    t1_pct   = (forecast[0] - current_price) / current_price * 100 if current_price else 0
    t2_pct   = (forecast[1] - forecast[0])   / forecast[0]   * 100 if forecast[0]   else 0

    def trend(pct):
        arrow = "▲" if pct >= 0 else "▼"
        cls   = "trend-up" if pct >= 0 else "trend-down"
        return f'<div class="{cls}">{arrow} {abs(pct):.1f}% vs prior</div>'

    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">�</div>
            <div class="card-title">Step 4 — Price Forecast</div>
        </div>
        <div class="stats-row" style="margin-bottom:1rem;">
            <div class="stat-chip"><div class="stat-label">Current Price</div><div class="stat-value">₹{current_price:,.0f}</div></div>
            <div class="stat-chip"><div class="stat-label">Data Points</div><div class="stat-value">{len(prices)}</div></div>
            <div class="stat-chip"><div class="stat-label">Horizon</div><div class="stat-value">2 months</div></div>
        </div>
        <div class="price-grid">
            <div class="price-box">
                <div class="price-month">Next Month</div>
                <div class="price-value">₹{forecast[0]:,.0f}</div>
                <div class="price-unit">per quintal (INR)</div>
                {trend(t1_pct)}
            </div>
            <div class="price-box">
                <div class="price-month">Month After</div>
                <div class="price-value">₹{forecast[1]:,.0f}</div>
                <div class="price-unit">per quintal (INR)</div>
                {trend(t2_pct)}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>AgroVision &nbsp;·&nbsp; ResNet50 + LSTM &nbsp;·&nbsp; Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
