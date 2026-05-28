```python
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from pipelines.fusion_inference import predict_fusion
from pipelines.food_gate import is_food_image

from pipelines.explain_cam import SimpleGradCAM, overlay_cam
from pipelines.image_encoder import (
    get_encoder_model,
    get_target_layer,
    get_device,
    get_cam_transform
)

from nutrition.nutrient_postprocess import compute_akg_percent
from nutrition.recommendation_engine import build_recommendation
from utils.report_engine import build_pdf_report


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="SENDE — Sistem Evaluasi Nilai Gizi dan Diet Elektronik MPASI",
    layout="wide"
)

# =====================================================
# STYLE
# =====================================================

st.markdown("""
<style>
h1 { text-align:center; }

div.stButton > button {
    background-color: #FF8C42 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}

div.stDownloadButton > button {
    background-color: #FF8C42 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] {
    background: #CFE8D5;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# TITLE
# =====================================================

st.title("🥣 SENDE — Sistem Evaluasi Nilai Gizi dan Diet Elektronik MPASI")
st.caption("Human-Centered Multimodal AI + Explainable Decision Support")


# =====================================================
# LOGIN
# =====================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.subheader("Informasi Pengguna")

    u = st.text_input("Nama User")
    r = st.selectbox("Peran Pengguna", ["Orang Tua", "Petugas Kesehatan"])

    if st.button("Masuk ke Sistem"):
        st.session_state.logged_in = True
        st.session_state.user = u
        st.session_state.role = r
        st.rerun()

    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Profil Anak")

infant_name = st.sidebar.text_input("Nama")
age_months = st.sidebar.number_input("Umur (bulan)", 6, 24, 8)
sex = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
weight = st.sidebar.number_input("Berat (kg)", 4.0, 20.0, 8.0)
height = st.sidebar.number_input("Tinggi (cm)", 55, 100, 70)

st.sidebar.caption(
    "Persentase kecukupan gizi dibanding kebutuhan harian AKG"
)


# =====================================================
# HISTORY
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =====================================================
# AGE GROUP
# =====================================================

def map_akg_group(age):
    if age <= 8:
        return "6-9_bulan"
    elif age <= 11:
        return "9-11_bulan"
    else:
        return "12-23_bulan"


age_group = map_akg_group(age_months)


# =====================================================
# NUTRIENT RANGE
# =====================================================

RANGE = {
    "Energi (kkal)": (110, 260),
    "Karbo (g)": (8, 30),
    "Lemak (g)": (4, 12),
    "Protein (g)": (5, 14),
    "Kalsium (mg)": (20, 80),
    "Vit A (RE)": (100, 600),
    "Vit C (mg)": (1, 10),
    "Vit E (mcg)": (0.1, 3),
    "Zinc (mg)": (0.1, 5),
    "Zat Besi (mg)": (0.1, 4)
}

DISPLAY_NAME = {
    "Energi (kkal)": "Energi (kkal)",
    "Karbo (g)": "Karbohidrat (g)",
    "Lemak (g)": "Lemak (g)",
    "Protein (g)": "Protein (g)",
    "Kalsium (mg)": "Kalsium (mg)",
    "Vit A (RE)": "Vitamin A (RE)",
    "Vit C (mg)": "Vitamin C (mg)",
    "Vit E (mcg)": "Vitamin E (mcg)",
    "Zinc (mg)": "Zinc (mg)",
    "Zat Besi (mg)": "Zat Besi (mg)"
}

NUT_KEYS = list(RANGE.keys())


# =====================================================
# HELPERS
# =====================================================

def vector_to_dict(vec):
    return {k: float(vec[i]) for i, k in enumerate(NUT_KEYS)}


def calibrated_output(n):
    out = {}

    for k, v in n.items():

        lo, hi = RANGE[k]

        # sigmoid calibration
        x = 1 / (1 + np.exp(-v))

        out[k] = round(lo + x * (hi - lo), 2)

    return out


def compute_confidence(vec):

    spread = np.std(vec)

    return round(
        float(1 / (1 + np.exp(-spread * 3))) * 100,
        1
    )


# =====================================================
# VISUALIZATION
# =====================================================

def plot_radar(nutrisi):

    labels = [DISPLAY_NAME.get(k, k) for k in nutrisi.keys()]
    values = list(nutrisi.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself'
    ))

    fig.update_layout(
        height=360,
        showlegend=False
    )

    return fig


def plot_bar(akg):

    display_vals = []

    for v in akg.values():

        # visual cap supaya realistis
        capped = min(v, 150)

        display_vals.append(capped)

    df = pd.DataFrame({
        "Nutrient": [DISPLAY_NAME.get(k, k) for k in akg.keys()],
        "AKG %": display_vals
    })

    fig = px.bar(
        df,
        x="Nutrient",
        y="AKG %",
        text="AKG %",
        height=360
    )

    fig.update_layout(
        yaxis_title="Persentase AKG (%)",
        xaxis_title=""
    )

    return fig


def nutrition_status(p):

    if p < 70:
        return "Kurang"

    elif p <= 130:
        return "Sesuai"

    else:
        return "Padat Nutrisi"


def translate_rec_text(text):

    for indo, eng in DISPLAY_NAME.items():
        text = text.replace(indo, eng)

    return text


# =====================================================
# UPLOAD
# =====================================================

uploaded = st.file_uploader(
    "Upload complementary food image",
    ["jpg", "jpeg", "png"]
)

if uploaded:

    img = Image.open(uploaded).convert("RGB")

    img_np = np.array(img)

    img_display = img.resize((224, 224))

    st.image(
        img_display,
        caption="Preview (224x224)",
        use_container_width=False
    )

    if st.button("Analisis Gizi", use_container_width=True):

        # =================================================
        # FOOD GATE
        # =================================================

        if not is_food_image(img):

            st.error("Citra terdeteksi bukan makanan")

            st.stop()

        # =================================================
        # MULTIMODAL INFERENCE
        # =================================================

        vec = predict_fusion(img)

        nutrisi = calibrated_output(
            vector_to_dict(vec)
        )

        conf = compute_confidence(vec)

        # =================================================
        # AKG PERCENT
        # =================================================

        # FINAL FIX:
        # tidak lagi divide /0.33

        akg = compute_akg_percent(
            nutrisi,
            age_group
        )

        # =================================================
        # GRADCAM
        # =================================================

        cam_model = SimpleGradCAM(
            get_encoder_model(),
            get_target_layer()
        )

        tensor = get_cam_transform()(img).unsqueeze(0).to(get_device())

        cam = cam_model.generate(tensor)

        cam_img = overlay_cam(img_np, cam)

        # =================================================
        # SAVE SESSION
        # =================================================

        st.session_state.nutrisi = nutrisi
        st.session_state.akg = akg
        st.session_state.conf = conf
        st.session_state.img = img
        st.session_state.cam = cam_img

        st.session_state.history.append({

            "label": datetime.now().strftime("%H:%M:%S"),

            "nutrisi": nutrisi,

            "akg": akg,

            "conf": conf,

            "img": img,

            "cam": cam_img
        })


# =====================================================
# DASHBOARD
# =====================================================

if "nutrisi" in st.session_state:

    nutrisi = st.session_state.nutrisi
    akg = st.session_state.akg

    st.subheader("Informasi Anak")

    cA, cB, cC = st.columns(3)

    cA.metric("Nama", infant_name or "-")
    cB.metric("Umur", f"{age_months} months")
    cC.metric("Jenis Kelamin", sex)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("Input Citra")

        st.image(
            st.session_state.img,
            use_container_width=True
        )

    with c2:

        st.subheader("GradCAM")

        st.image(
            st.session_state.cam,
            use_container_width=True
        )

    with c3:

        st.subheader("Estimasi Kandungan Gizi")

        st.metric(
            "Model Confidence",
            f"{st.session_state.conf}%"
        )

        for k, v in nutrisi.items():

            st.write(
                DISPLAY_NAME.get(k, k),
                v
            )

    st.plotly_chart(
        plot_radar(nutrisi),
        use_container_width=True
    )

    st.plotly_chart(
        plot_bar(akg),
        use_container_width=True
    )

    # =================================================
    # STATUS
    # =================================================

    st.subheader("Status Kecukupan Gizi")

    for k, v in akg.items():

        st.write(
            f"{DISPLAY_NAME.get(k, k)} : "
            f"{round(v,1)}% "
            f"({nutrition_status(v)})"
        )

    # =================================================
    # RECOMMENDATION
    # =================================================

    st.subheader("Rekomendasi Gizi Personal")

    recs = build_recommendation(akg)

    for r in recs:

        st.write(
            "•",
            translate_rec_text(r)
        )

    st.caption(
        "Persentase AKG digunakan sebagai "
        "indikator interpretatif berbasis estimasi AI "
        "dan bukan evaluasi klinis harian penuh."
    )

    # =================================================
    # PDF
    # =================================================

    if st.button("Buat Laporan PDF"):

        child = {
            "name": infant_name,
            "age_months": age_months,
            "sex": sex,
            "weight": weight,
            "height": height
        }

        path = build_pdf_report(
            child,
            nutrisi,
            akg,
            recs
        )

        with open(path, "rb") as f:

            st.download_button(
                "Download PDF",
                f,
                "sende_report.pdf"
            )


# =====================================================
# HISTORY
# =====================================================

st.sidebar.divider()

st.sidebar.subheader("Analysis History")

if st.session_state.history:

    labels = [h["label"] for h in st.session_state.history]

    sel = st.sidebar.selectbox(
        "Open previous result",
        labels
    )

    if st.sidebar.button("Load Selected"):

        h = next(
            x for x in st.session_state.history
            if x["label"] == sel
        )

        st.session_state.nutrisi = h["nutrisi"]
        st.session_state.akg = h["akg"]
        st.session_state.conf = h["conf"]
        st.session_state.img = h["img"]
        st.session_state.cam = h["cam"]

        st.rerun()
```
