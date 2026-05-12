"""
Land Cover AI — Streamlit Inference App
Point-Supervised U-Net Semantic Segmentation on Aerial Imagery
"""

import io
import os
import warnings

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "30pts.pth")
NUM_CLASSES = 5
IMAGE_SIZE  = 512          # tile size used during training
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Land-cover class definitions  (index → label, colour)
CLASS_INFO = {
    0: {"label": "Background",   "color": (30,  30,  30)},
    1: {"label": "Building",     "color": (220, 20,  60)},
    2: {"label": "Woodland",     "color": (34, 139,  34)},
    3: {"label": "Water",        "color": (30, 144, 255)},
    4: {"label": "Road",         "color": (255, 165,  0)},
}


# ─── Model Definition (must match training exactly) ───────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for feature in features:
            self.encoder_blocks.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))

        self.output = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for idx in range(len(self.decoder_blocks)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[idx](x)

        return self.output(x)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(path: str) -> UNet:
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """Resize → float32 [0,1] → CHW tensor → batch dim."""
    img = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0).to(DEVICE)            # [1, 3, H, W]


def predict(model: UNet, tensor: torch.Tensor) -> np.ndarray:
    """Run inference → class-index map [H, W]."""
    with torch.no_grad():
        logits = model(tensor)                        # [1, C, H, W]
    pred = torch.argmax(logits, dim=1).squeeze(0)    # [H, W]
    return pred.cpu().numpy().astype(np.uint8)


def colorise(pred_map: np.ndarray) -> np.ndarray:
    """Map class indices → RGB colour image."""
    h, w = pred_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, info in CLASS_INFO.items():
        mask = pred_map == cls_idx
        rgb[mask] = info["color"]
    return rgb


def class_stats(pred_map: np.ndarray) -> dict:
    total = pred_map.size
    stats = {}
    for cls_idx, info in CLASS_INFO.items():
        count = int((pred_map == cls_idx).sum())
        stats[info["label"]] = round(count / total * 100, 2)
    return stats


def overlay(original_bgr: np.ndarray, colour_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    orig_rgb = cv2.cvtColor(
        cv2.resize(original_bgr, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB
    )
    blended = (orig_rgb * (1 - alpha) + colour_mask * alpha).astype(np.uint8)
    return blended


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Land Cover AI · Segmentation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 50%, #e0e7ff 100%);
    border-radius: 18px;
    padding: 3rem 4rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(186, 230, 253, 0.4);
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
    text-align: center;
  }
  .hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(circle at center, rgba(255,255,255,0.8) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero h1 {
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(90deg, #2563eb, #06b6d4, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.8rem;
  }
  .hero p  { color: #475569; font-size: 1.15rem; line-height: 1.7; font-weight: 500; }

  /* Cards */
  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
  }
  
  .feature-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #2563eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
    height: 100%;
  }

  /* Legend pills */
  .legend-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
    color: #fff;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  }

  /* Metric tiles */
  .metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02);
  }
  .metric-box .val { font-size: 1.8rem; font-weight: 700; color: #2563eb; }
  .metric-box .lbl { font-size: 0.85rem; color: #64748b; margin-top: 4px; font-weight: 500; }

  /* Upload area */
  [data-testid="stFileUploader"] {
    border: 2px dashed #93c5fd !important;
    border-radius: 16px;
    background: #eff6ff;
    transition: all 0.25s;
    padding: 2rem;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
    background: #e0f2fe;
  }

  /* Image captions */
  .caption {
    text-align: center; color: #64748b;
    font-size: 0.85rem; margin-top: 8px;
    font-weight: 500;
  }
  
  /* Make tabs look like a navbar */
  button[data-baseweb="tab"] {
    font-size: 1.1rem !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
  }
  
  /* Sidebar */
  [data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 1.8rem;
    font-weight: 600; font-size: 1rem;
    transition: transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 4px 16px rgba(37,99,235,0.25);
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37,99,235,0.35);
    color: white;
  }
  
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    overlay_alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05,
                              help="Blend ratio of colour mask over original image")
    show_overlay  = st.toggle("Show blended overlay", value=True)
    st.divider()

    st.markdown("### 🗺️ Class Legend")
    for info in CLASS_INFO.values():
        r, g, b = info["color"]
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        st.markdown(
            f'<span class="legend-pill" style="background:{hex_col};">'
            f'{info["label"]}</span>',
            unsafe_allow_html=True,
        )
    st.divider()

    st.markdown("### ℹ️ Model Info")
    st.markdown(
        '<div class="card" style="padding: 1.2rem;">'
        "<b>Architecture:</b> U-Net (custom)<br>"
        f"<b>Classes:</b> {NUM_CLASSES}<br>"
        f"<b>Tile size:</b> {IMAGE_SIZE}×{IMAGE_SIZE} px<br>"
        "<b>Loss:</b> Partial Focal Loss<br>"
        f"<b>Device:</b> {DEVICE.upper()}"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Navigation Navbar ────────────────────────────────────────────────────────
tab_home, tab_predict, tab_feature, tab_dev = st.tabs([
    "🏠 Home", "🔍 Predict", "✨ Feature", "👨‍💻 Developer"
])

with tab_home:
    # ─── Hero Banner ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>🛰️ Land Cover AI</h1>
      <p>
        Point-supervised semantic segmentation of high-resolution aerial imagery using a custom U-Net.<br>
        Upload a <b>.tif</b>, <b>.jpg</b>, or <b>.png</b> aerial tile and get instant land-cover predictions.
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #0f172a; margin-top: 0;">Welcome to Land Cover AI</h3>
        <p>This application uses deep learning to segment high-resolution aerial images into five distinct classes: Background, Building, Woodland, Water, and Road.</p>
        <p>Navigate to the <b>Predict</b> tab to try it out, or check the <b>Feature</b> tab to learn more about the technology behind it.</p>
    </div>
    """, unsafe_allow_html=True)

with tab_feature:
    st.markdown("""
    <div class="card">
        <h2 style="color: #0f172a; margin-top: 0;">✨ Key Features</h2>
        <p>Discover the powerful capabilities of our Land Cover AI system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #1e293b; margin-top: 0;">🧠 Custom U-Net Architecture</h4>
            <p style="color: #475569;">Utilizes a deep convolutional network designed specifically for precise image segmentation tasks, ensuring high spatial accuracy across various terrains.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #1e293b; margin-top: 0;">📍 Point-Supervised Learning</h4>
            <p style="color: #475569;">Trained using a highly efficient point-supervision strategy, allowing the model to generalize well even with sparse annotations during the training phase.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #1e293b; margin-top: 0;">⚡ Real-Time Processing</h4>
            <p style="color: #475569;">Runs inference efficiently using PyTorch, providing fast visual feedback and detailed masking on your uploaded aerial images directly in the browser.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #1e293b; margin-top: 0;">📊 Interactive Analytics</h4>
            <p style="color: #475569;">Generates detailed coverage statistics instantly, helping you analyze the exact percentage of land use across different categories in your area of interest.</p>
        </div>
        """, unsafe_allow_html=True)

with tab_dev:
    st.markdown("""
    <div class="card" style="text-align: center; padding: 4rem 2rem;">
        <h2 style="color: #0f172a; margin-top: 0;">👨‍💻 About the Developer</h2>
        <h1 style="color: #2563eb; font-weight: 800; font-size: 2.5rem; margin-top: 1rem;">Shaurya</h1>
        <p style="font-size: 1.2rem; color: #475569; margin-top: 1.5rem; max-width: 600px; margin-left: auto; margin-right: auto;">
            A passionate software engineer focused on applied AI, computer vision, and building impactful digital experiences.
        </p>
        <hr style="margin: 3rem auto; opacity: 0.5; max-width: 200px; border-color: #cbd5e1;">
        <p style="color: #64748b; font-weight: 500;">
            This project demonstrates the deployment of a custom PyTorch semantic segmentation model wrapped in a sleek, interactive Streamlit frontend.
        </p>
    </div>
    """, unsafe_allow_html=True)


with tab_predict:
    # ─── Load model (cached) ─────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model weights not found at **{MODEL_PATH}**.\n\n"
            "Make sure `best_model_30pts_Focal.pth` is in the project root."
        )
        st.stop()

    model = load_model(MODEL_PATH)


    # ─── File uploader ────────────────────────────────────────────────────────────
    col_up, col_spacer = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drop an aerial image here or click to browse",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
            label_visibility="visible",
        )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center; color:#64748b; padding:4rem 0; font-size:1.1rem; background: #f8fafc; border-radius: 16px; border: 2px dashed #cbd5e1; margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⬆️</div>
            Upload an aerial image to start prediction
        </div>
        """, unsafe_allow_html=True)
    else:
        # ─── Read & validate image ────────────────────────────────────────────────────
        raw_bytes = uploaded.read()
        file_arr  = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr   = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            # Fallback: PIL (handles some .tif variants)
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                st.error(f"Could not decode image: {exc}")
                st.stop()

        orig_h, orig_w = img_bgr.shape[:2]


        # ─── Run prediction ───────────────────────────────────────────────────────────
        with st.spinner("🔍 Running segmentation…"):
            tensor       = preprocess(img_bgr)
            pred_map     = predict(model, tensor)
            colour_mask  = colorise(pred_map)
            blended      = overlay(img_bgr, colour_mask, alpha=overlay_alpha)
            orig_rgb     = cv2.cvtColor(
                cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB
            )
            stats        = class_stats(pred_map)


        # ─── Metrics row ──────────────────────────────────────────────────────────────
        st.markdown("### 📊 Coverage Statistics")
        metric_cols = st.columns(NUM_CLASSES)
        for i, (label, pct) in enumerate(stats.items()):
            with metric_cols[i]:
                r, g, b = list(CLASS_INFO.values())[i]["color"]
                hex_col  = f"#{r:02x}{g:02x}{b:02x}"
                st.markdown(
                    f'<div class="metric-box" style="border-bottom: 4px solid {hex_col};">'
                    f'<div class="val" style="color:{hex_col};">{pct}%</div>'
                    f'<div class="lbl">{label}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.divider()


        # ─── Image panels ─────────────────────────────────────────────────────────────
        if show_overlay:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(orig_rgb, use_container_width=True)
                st.markdown('<p class="caption">Original (resized)</p>', unsafe_allow_html=True)
            with c2:
                st.image(colour_mask, use_container_width=True)
                st.markdown('<p class="caption">Segmentation Mask</p>', unsafe_allow_html=True)
            with c3:
                st.image(blended, use_container_width=True)
                st.markdown('<p class="caption">Blended Overlay</p>', unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(orig_rgb, use_container_width=True)
                st.markdown('<p class="caption">Original (resized)</p>', unsafe_allow_html=True)
            with c2:
                st.image(colour_mask, use_container_width=True)
                st.markdown('<p class="caption">Segmentation Mask</p>', unsafe_allow_html=True)


        # ─── Image metadata ───────────────────────────────────────────────────────────
        st.divider()
        with st.expander("🔎 Image metadata"):
            meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
            meta_c1.metric("Filename",     uploaded.name)
            meta_c2.metric("Original size", f"{orig_w} × {orig_h}")
            meta_c3.metric("File size",    f"{len(raw_bytes)/1024:.1f} KB")
            meta_c4.metric("Model input",  f"{IMAGE_SIZE} × {IMAGE_SIZE}")


        # ─── Download mask ────────────────────────────────────────────────────────────
        mask_pil     = Image.fromarray(colour_mask)
        buf          = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️  Download segmentation mask (PNG)",
            data=buf.getvalue(),
            file_name=f"{os.path.splitext(uploaded.name)[0]}_mask.png",
            mime="image/png",
        )
