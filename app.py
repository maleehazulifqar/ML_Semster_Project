import base64
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "scratches",
]

CLASS_DESCRIPTIONS = {
    "Crazing": "Fine hairline crack network across the rolled steel surface.",
    "Inclusion": "Non-metallic foreign particles embedded beneath or on the surface.",
    "Patches": "Irregular localized surface discoloration or film patches.",
    "Pitted_Surface": "Small cavities or pits formed on the metal surface.",
    "Rolled-in_Scale": "Oxide scale pressed into the surface during hot rolling.",
    "scratches": "Linear mechanical abrasion marks along or across the strip.",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dual_head_resnet50.keras")
DL_RESULTS_PATH = os.path.join(BASE_DIR, "results", "pretrained_dual_head_results.json")
BASELINE_CSV_PATH = os.path.join(BASE_DIR, "results", "baseline_summary.csv")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


def find_asset_file(base_name: str) -> str | None:
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".jpg.jpg", ".png.png"]:
        candidate = os.path.join(ASSETS_DIR, f"{base_name}{ext}")
        if os.path.isfile(candidate):
            return candidate
    if os.path.isdir(ASSETS_DIR):
        for fname in os.listdir(ASSETS_DIR):
            stem, _ = os.path.splitext(fname)
            if stem.lower() == base_name.lower():
                full = os.path.join(ASSETS_DIR, fname)
                if os.path.isfile(full):
                    return full
    return None


def get_base64_image(base_name: str) -> tuple[str | None, str]:
    image_path = find_asset_file(base_name)
    if not image_path:
        return None, "image/jpeg"
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if "png" in ext else ("image/webp" if "webp" in ext else "image/jpeg")
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime
    except Exception:
        return None, mime


def apply_styles():
    st.set_page_config(
        page_title="NEU Surface Defect & Severity Inspector",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    app_bg_b64, app_mime = get_base64_image("app_bg")
    banner_bg_b64, banner_mime = get_base64_image("banner_bg")

    if app_bg_b64:
        app_bg_css = f"""
            background: linear-gradient(rgba(5, 7, 4, 0.80), rgba(0, 0, 0, 0.88)),
                        url("data:{app_mime};base64,{app_bg_b64}") center center / cover fixed no-repeat;
        """
    else:
        app_bg_css = """
            background: linear-gradient(180deg, #070905 0%, #000000 100%);
        """

    if banner_bg_b64:
        banner_bg_css = f"""
            background: linear-gradient(100deg, rgba(6, 8, 5, 0.84) 0%, rgba(12, 16, 9, 0.56) 52%, rgba(6, 8, 5, 0.78) 100%),
                        url("data:{banner_mime};base64,{banner_bg_b64}") center center / cover no-repeat;
        """
    else:
        banner_bg_css = """
            background: linear-gradient(125deg, #14190E 0%, #090B07 65%, #1D2212 100%);
        """

    st.markdown(
        f"""
        <style>
        /* Global Times New Roman EXCEPT Streamlit Material Icons */
        html, body, .stApp, p,
        span:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded):not(.material-icons),
        div:not([data-testid="stIconMaterial"]),
        label, input, button, select, textarea, table, th, td {{
            font-family: "Times New Roman", Times, Georgia, serif !important;
        }}

        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        .material-icons {{
            font-family: "Material Symbols Rounded", "Material Icons" !important;
            overflow: visible !important;
            white-space: nowrap !important;
        }}

        .stApp {{
            {app_bg_css}
            color: #F5EFD6 !important;
        }}

        h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            font-family: "Times New Roman", Times, Georgia, serif !important;
            color: #F5EFD6 !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }}

        header[data-testid="stHeader"] {{
            background-color: #000000 !important;
            border-bottom: 1px solid #636B2F !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: #0C0F09 !important;
            border-right: 1.5px solid #636B2F !important;
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]),
        section[data-testid="stSidebar"] div:not([data-testid="stIconMaterial"]),
        section[data-testid="stSidebar"] .stMarkdown {{
            color: #F5EFD6 !important;
            opacity: 1 !important;
        }}

        div[data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1.5px solid #636B2F !important;
            padding-bottom: 4px;
        }}

        button[data-baseweb="tab"] {{
            background-color: #14190E !important;
            border: 1px solid rgba(245, 239, 214, 0.35) !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 22px !important;
            opacity: 1 !important;
        }}

        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span,
        button[data-baseweb="tab"] div {{
            color: #F5EFD6 !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            background-color: #636B2F !important;
            border: 1.5px solid #F5EFD6 !important;
        }}

        .top-banner {{
            {banner_bg_css}
            border: 1.5px solid #636B2F;
            border-radius: 12px;
            padding: 32px 34px;
            margin-bottom: 20px;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.60);
        }}

        .top-tag {{
            display: inline-block;
            background-color: #636B2F;
            color: #F5EFD6;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .top-title {{
            margin: 0;
            font-size: 2.15rem;
            color: #F5EFD6;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.95);
        }}

        .top-sub {{
            color: #F5EFD6;
            font-size: 1.08rem;
            line-height: 1.55;
            margin: 8px 0 0 0;
            max-width: 860px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.95);
        }}

        .kpi-card {{
            background-color: rgba(13, 16, 10, 0.92);
            border: 1px solid rgba(245, 239, 214, 0.24);
            border-top: 3.5px solid #636B2F;
            border-radius: 10px;
            padding: 16px 18px;
        }}

        .kpi-title {{
            color: #D6CCA8;
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            margin-bottom: 6px;
        }}

        .kpi-num {{
            color: #F5EFD6;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.15;
        }}

        .kpi-note {{
            color: #C4B994;
            font-size: 0.92rem;
            margin-top: 6px;
        }}

        .meter-track {{
            width: 100%;
            height: 14px;
            background-color: #1B2014;
            border: 1px solid rgba(245, 239, 214, 0.25);
            border-radius: 6px;
            overflow: hidden;
            margin: 10px 0 14px 0;
        }}

        .meter-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        .status-box {{
            background-color: #14190E;
            border-left: 4px solid #636B2F;
            border-top: 1px solid rgba(245, 239, 214, 0.18);
            border-right: 1px solid rgba(245, 239, 214, 0.18);
            border-bottom: 1px solid rgba(245, 239, 214, 0.18);
            color: #F5EFD6;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 1.0rem;
        }}

        div[data-testid="stAlert"] {{
            background-color: #EFF6FF !important;
            border: 1px solid #BFDBFE !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
        }}

        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span,
        div[data-testid="stAlert"] div {{
            color: #1D4ED8 !important;
            font-size: 1.08rem !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }}

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {{
            color: #F5EFD6 !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background-color: #11150D !important;
            border: 1.5px dashed #636B2F !important;
            border-radius: 10px !important;
            padding: 16px !important;
        }}

        [data-testid="stFileUploaderDropzone"] span:not([data-testid="stIconMaterial"]),
        [data-testid="stFileUploaderDropzone"] small {{
            color: #E6DEC0 !important;
            opacity: 1 !important;
        }}

        [data-testid="stFileUploaderDropzone"] button {{
            background-color: #636B2F !important;
            color: #F5EFD6 !important;
            border: 1px solid #F5EFD6 !important;
            border-radius: 6px !important;
            padding: 6px 16px !important;
            font-weight: 700 !important;
        }}

        [data-testid="stFileUploaderDropzone"] button *:not([data-testid="stIconMaterial"]) {{
            color: #F5EFD6 !important;
            opacity: 1 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading ResNet50 Dual-Head model...")
def load_dual_head_model(model_path: str):
    if not os.path.exists(model_path):
        return None, f"Model file not found at: {model_path}"
    try:
        m = tf.keras.models.load_model(model_path, compile=False)
        return m, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_benchmark_data():
    dl_data = None
    baseline_df = None

    if os.path.exists(DL_RESULTS_PATH):
        try:
            with open(DL_RESULTS_PATH, "r", encoding="utf-8") as f:
                dl_data = json.load(f)
        except Exception:
            pass

    if os.path.exists(BASELINE_CSV_PATH):
        try:
            baseline_df = pd.read_csv(BASELINE_CSV_PATH)
        except Exception:
            pass

    return dl_data, baseline_df


def preprocess_uploaded_image(pil_img: Image.Image, target_size=(224, 224)) -> tuple[np.ndarray, np.ndarray]:
    rgb_img = pil_img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    arr = np.asarray(rgb_img, dtype=np.float32)
    batch = np.expand_dims(arr.copy(), axis=0)
    batch = resnet_preprocess(batch)
    return np.asarray(rgb_img), batch


def evaluate_severity_level(score: float, warn_thresh: float, crit_thresh: float):
    score = float(np.clip(score, 0.0, 1.0))
    if score >= crit_thresh:
        return (
            "CRITICAL — REJECT / HOLD",
            "#D9534F",
            "Immediate coil hold or rejection advised; escalate to quality supervisor.",
        )
    if score >= warn_thresh:
        return (
            "MODERATE — SECONDARY CHECK",
            "#D4A017",
            "Route sample for secondary inspection and rolling parameter review.",
        )
    return (
        "LOW — ACCEPTABLE",
        "#8FA343",
        "Severity within acceptable operating range; record inspection log.",
    )


def render_banner():
    st.markdown(
        """
        <div class="top-banner">
            <div class="top-tag">NEU-DET Quality Inspection Suite</div>
            <h1 class="top-title">
                Industrial Surface Defect &amp; Severity Inspector
            </h1>
            <p class="top-sub">
                Dual-Head ResNet50 inspection system for six hot-rolled steel defect classes paired with
                continuous severity scoring (0.00–1.00) and classical ML baseline benchmarks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(dl_data: dict | None, baseline_df: pd.DataFrame | None):
    c1, c2, c3, c4 = st.columns(4)

    dl_acc = "—"
    dl_f1 = "—"
    dl_r2 = "—"
    if dl_data and "test" in dl_data:
        dl_acc = f"{dl_data['test'].get('accuracy', 0.0) * 100:.2f}%"
        dl_f1 = f"{dl_data['test'].get('f1', 0.0):.4f}"
        dl_r2 = f"{dl_data['test'].get('r2', 0.0):.4f}"

    best_base = "—"
    if baseline_df is not None and not baseline_df.empty:
        test_rows = baseline_df[baseline_df["split"] == "test"]
        if not test_rows.empty:
            best_base = f"{test_rows['accuracy'].max() * 100:.2f}%"

    cards = [
        (c1, "DL Test Accuracy", dl_acc, "ResNet50 Dual-Head (270 test samples)"),
        (c2, "DL Test F1-Score", dl_f1, "6-class weighted classification"),
        (c3, "DL Severity R²", dl_r2, "Regression score [0.00 – 1.00]"),
        (c4, "Best Baseline Acc", best_base, "Classical ML test benchmark"),
    ]

    for col, title, val, note in cards:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-num">{val}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_inspector(model, model_err: str | None, warn_thresh: float, crit_thresh: float):
    col_left, col_right = st.columns([1.05, 1.25], gap="large")

    disp_img = None
    batch = None
    img_info = ""
    upload_err = None

    with col_left:
        st.subheader("Sample Image Input")
        uploaded = st.file_uploader(
            "Select a surface defect image (PNG, JPG, JPEG, BMP)",
            type=["png", "jpg", "jpeg", "bmp"],
        )

        if uploaded is None:
            st.info("Upload a steel surface image to inspect defect class and severity score.")
            st.markdown("**NEU-DET Defect Categories:**")
            for idx, cname in enumerate(CLASS_NAMES):
                st.markdown(f"- **{idx}. {cname}** — {CLASS_DESCRIPTIONS[cname]}")
        else:
            try:
                pil_img = Image.open(uploaded)
                disp_img, batch = preprocess_uploaded_image(pil_img, target_size=(224, 224))
                img_info = f"File: {uploaded.name} | Resized Preview (224×224) | Original: {pil_img.size[0]}×{pil_img.size[1]}"
                st.success(f"Image loaded: {uploaded.name}")
                st.image(disp_img, caption=img_info, use_container_width=True)
            except Exception as e:
                upload_err = str(e)
                st.error(f"Could not process image: {upload_err}")

    with col_right:
        st.subheader("Inspection Results")

        if uploaded is None:
            st.info("Waiting for an uploaded steel surface image...")
            return

        if upload_err:
            st.error(f"Cannot run prediction because image failed to load: {upload_err}")
            return

        if model is None:
            st.error(f"Model could not be loaded: {model_err or MODEL_PATH}")
            return

        try:
            preds = model.predict(batch, verbose=0)
            if isinstance(preds, dict):
                probs = np.asarray(preds["classification"]).squeeze()
                sev_val = float(np.asarray(preds["regression"]).squeeze())
            elif isinstance(preds, (list, tuple)) and len(preds) >= 2:
                probs = np.asarray(preds[0]).squeeze()
                sev_val = float(np.asarray(preds[1]).squeeze())
            else:
                st.error(f"Unexpected model output type: {type(preds)}")
                return

            probs = np.atleast_1d(np.asarray(probs, dtype=np.float32))
            if probs.min() < 0.0 or not np.isclose(probs.sum(), 1.0, atol=1e-2):
                probs = tf.nn.softmax(probs).numpy()

            top_idx = int(np.argmax(probs))
            top_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class_{top_idx}"
            top_conf = float(probs[top_idx]) * 100.0
            sev_clipped = float(np.clip(sev_val, 0.0, 1.0))

            sev_status, bar_color, recommendation = evaluate_severity_level(
                sev_clipped, warn_thresh, crit_thresh
            )

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Predicted Defect</div>
                        <div class="kpi-num" style="font-size:1.5rem;">{top_class}</div>
                        <div class="kpi-note">Confidence: <strong>{top_conf:.2f}%</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Severity Score</div>
                        <div class="kpi-num">{sev_clipped:.4f}</div>
                        <div class="kpi-note"><strong>{sev_status}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="meter-track">
                    <div class="meter-fill" style="width:{sev_clipped * 100:.1f}%; background-color:{bar_color};"></div>
                </div>
                <div class="status-box">
                    <strong>Action Guidance:</strong> {recommendation}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Class Probability Breakdown (%)")
            chart_df = pd.DataFrame(
                {
                    "Defect Class": CLASS_NAMES[: len(probs)],
                    "Probability (%)": np.round(probs[: len(CLASS_NAMES)] * 100.0, 2),
                }
            ).set_index("Defect Class")
            st.bar_chart(chart_df, color="#8FA343", height=220)

        except Exception as e:
            st.error(f"Prediction error: {e}")


def render_benchmarks(dl_data: dict | None, baseline_df: pd.DataFrame | None):
    st.subheader("Model Benchmark Comparison")

    rows = []
    if baseline_df is not None and not baseline_df.empty:
        for _, r in baseline_df.iterrows():
            rows.append(
                {
                    "Model": f"Classical ({r['features']})",
                    "Split": r["split"],
                    "Accuracy": round(float(r["accuracy"]), 4),
                    "F1-Score": round(float(r["f1"]), 4),
                    "MAE": round(float(r["mae"]), 4),
                    "MSE": round(float(r["mse"]), 5),
                    "R²": round(float(r["r2"]), 4),
                }
            )

    if dl_data:
        for split_key in ["val", "test"]:
            if split_key in dl_data:
                m = dl_data[split_key]
                rows.append(
                    {
                        "Model": dl_data.get("model", "ResNet50_DualHead"),
                        "Split": split_key,
                        "Accuracy": round(float(m["accuracy"]), 4),
                        "F1-Score": round(float(m["f1"]), 4),
                        "MAE": round(float(m["mae"]), 4),
                        "MSE": round(float(m["mse"]), 5),
                        "R²": round(float(m["r2"]), 4),
                    }
                )

    if rows:
        df = pd.DataFrame(rows)
        selected_split = st.selectbox("Filter Split", ["All", "test", "val", "cv_mean"], index=0)
        view_df = df if selected_split == "All" else df[df["Split"] == selected_split]
        st.dataframe(view_df, use_container_width=True, hide_index=True)
    else:
        st.info("Benchmark summary files not found in `results/`.")


def main():
    apply_styles()
    model, model_err = load_dual_head_model(MODEL_PATH)
    dl_data, baseline_df = load_benchmark_data()
    banner_found = find_asset_file("banner_bg") is not None

    with st.sidebar:
        st.markdown("### Inspector Controls")
        st.markdown("Adjust severity alert thresholds:")
        warn_thresh = st.slider("Moderate Threshold", 0.10, 0.70, 0.35, 0.05)
        crit_thresh = st.slider("Critical Threshold", 0.40, 0.95, 0.65, 0.05)
        if crit_thresh <= warn_thresh:
            crit_thresh = min(0.95, warn_thresh + 0.10)

        st.markdown("---")
        st.markdown("### System Status")
        st.markdown(f"- **ResNet50 Model:** {'Loaded' if model is not None else 'Missing'}")
        st.markdown(f"- **DL Metrics JSON:** {'Available' if dl_data is not None else 'Missing'}")
        st.markdown(f"- **Baseline CSV:** {'Available' if baseline_df is not None else 'Missing'}")
        st.markdown(f"- **Banner Image:** {'Found' if banner_found else 'Not Found in assets/'}")

    render_banner()
    render_kpis(dl_data, baseline_df)

    tab1, tab2 = st.tabs(["Live Defect Inspector", "Model Benchmarks"])
    with tab1:
        render_inspector(model, model_err, warn_thresh, crit_thresh)
    with tab2:
        render_benchmarks(dl_data, baseline_df)


if __name__ == "__main__":
    main()