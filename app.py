# ==========================================
# 🌸 SMART FLOWER IDENTIFIER
# CNN + TRANSFORMER + HYBRID AI
# ==========================================

import streamlit as st
import streamlit.components.v1 as components

from PIL import Image

import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np

import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌸 Smart Flower Identifier",
    layout="centered"
)

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#ede9fe,#fdf4ff);
}

/* TITLE */
.main-title {
    text-align:center;
    color:#6d28d9;
    font-size:45px;
    font-weight:bold;
    margin-bottom:10px;
}

/* TABLE */
.custom-table {
    width:100%;
    border-collapse:collapse;
    margin-top:10px;
    overflow:hidden;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

.custom-table th {
    background:#7c3aed;
    color:white;
    padding:12px;
    text-align:center;
    font-size:16px;
}

.custom-table td {
    padding:10px;
    text-align:center;
    font-weight:500;
}

.custom-table tr:nth-child(even) {
    background:#f3e8ff;
}

.custom-table tr:hover {
    background:#ddd6fe;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {
    background:rgba(255,255,255,0.9);
    padding:20px;
    border-radius:20px;
    border:2px dashed #a78bfa;
    box-shadow:0px 6px 15px rgba(0,0,0,0.08);
}

/* IMAGE */
img {
    border-radius:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<div class='main-title'>🌸 Smart Flower Identifier 🌸</div>",
    unsafe_allow_html=True
)

# ---------------- TABLE FUNCTION ----------------
def styled_table(df):

    html = "<table class='custom-table'>"

    html += "<tr>"

    for col in df.columns:
        html += f"<th>{col}</th>"

    html += "</tr>"

    for _, row in df.iterrows():

        html += "<tr>"

        for val in row:
            html += f"<td>{val}</td>"

        html += "</tr>"

    html += "</table>"

    return html

# ---------------- FLOWER INFO ----------------
flower_info = {

    "daisy": {
        "name":"Daisy 🌼",
        "scientific":"Bellis perennis",
        "meaning":"Innocence",
        "uses":"Gardens",
        "fun_fact":"Closes at night!"
    },

    "dandelion": {
        "name":"Dandelion 🌿",
        "scientific":"Taraxacum",
        "meaning":"Hope",
        "uses":"Medicine",
        "fun_fact":"All parts edible!"
    },

    "rose": {
        "name":"Rose 🌹",
        "scientific":"Rosa",
        "meaning":"Love",
        "uses":"Perfume",
        "fun_fact":"300+ species!"
    },

    "sunflower": {
        "name":"Sunflower 🌻",
        "scientific":"Helianthus",
        "meaning":"Happiness",
        "uses":"Oil",
        "fun_fact":"Follows the sun!"
    },

    "tulip": {
        "name":"Tulip 🌷",
        "scientific":"Tulipa",
        "meaning":"Love",
        "uses":"Decoration",
        "fun_fact":"Once cost more than gold!"
    }
}

# ---------------- LOAD CLASS NAMES ----------------
@st.cache_resource
def load_classes():

    return torch.load(
        "class_names.pth",
        map_location="cpu"
    )

# ---------------- LOAD CNN ----------------
@st.cache_resource
def load_cnn():

    from utils import load_model

    class_names = load_classes()

    model = load_model(
        "flower_model.pth",
        len(class_names)
    )

    model.eval()

    return model, class_names

# ---------------- LOAD TRANSFORMER ----------------
@st.cache_resource
def load_vit():

    from transformers import (
        ViTForImageClassification,
        AutoImageProcessor
    )

    class_names = load_classes()

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(class_names),
        ignore_mismatched_sizes=True
    )

    processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224"
    )

    model.eval()

    return model, processor, class_names

# ---------------- CNN PREDICTION ----------------
def predict_cnn(image, model, class_names):

    from utils import predict_image

    return predict_image(
        image,
        model,
        class_names
    )

# ---------------- TRANSFORMER PREDICTION ----------------
def predict_vit(image, model, processor, class_names):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probs = F.softmax(
        outputs.logits,
        dim=1
    )[0]

    confidence, pred = torch.max(
        probs,
        dim=0
    )

    return (
        class_names[pred.item()],
        confidence.item(),
        probs.numpy()
    )

# ---------------- HYBRID MODEL ----------------
def hybrid_predict(
    cnn_probs,
    vit_probs,
    class_names
):

    cnn_probs = np.array(cnn_probs)
    vit_probs = np.array(vit_probs)

    hybrid_probs = (
        cnn_probs + vit_probs
    ) / 2

    pred_index = np.argmax(hybrid_probs)

    return (
        class_names[pred_index],
        float(hybrid_probs[pred_index]),
        hybrid_probs
    )

# ---------------- SESSION ----------------
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

# ---------------- WELCOME UI ----------------
if not st.session_state.uploaded:

    components.html(
        """
        <div style="
            padding:35px;
            border-radius:25px;
            background:rgba(255,255,255,0.75);
            backdrop-filter:blur(10px);
            text-align:center;
            margin-top:30px;
            box-shadow:0px 8px 25px rgba(124,58,237,0.15);
            font-family:sans-serif;
        ">

            <h2 style="
                color:#7c3aed;
                font-size:32px;
            ">
            ✨ Upload a Flower Image ✨
            </h2>

            <p style="
                color:#6b7280;
                font-size:18px;
            ">
            Discover flowers using
            CNN + Transformer + Hybrid AI
            </p>

            <div style="
                font-size:42px;
                margin-top:15px;
            ">
            🌸 🌼 🌷 🌻 🌹
            </div>

        </div>
        """,
        height=300
    )

# ---------------- FILE UPLOADER ----------------
st.subheader("📷 Upload Flower Image")

file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

# ================= MAIN =================
if file:

    st.session_state.uploaded = True

    image = Image.open(file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Flower",
        width=300
    )

    with st.spinner("🔍 AI analyzing flower..."):

        time.sleep(1)

        cnn_model, class_names = load_cnn()

        vit_model, processor, _ = load_vit()

        # CNN
        cnn_pred, cnn_conf, cnn_probs = predict_cnn(
            image,
            cnn_model,
            class_names
        )

        # Transformer
        vit_pred, vit_conf, vit_probs = predict_vit(
            image,
            vit_model,
            processor,
            class_names
        )

        # Hybrid
        hybrid_pred, hybrid_conf, hybrid_probs = hybrid_predict(
            cnn_probs,
            vit_probs,
            class_names
        )

    # ================= MODEL CARDS =================
    st.markdown("## 🔍 Model Predictions")

    col1, col2, col3 = st.columns(3)

    # CNN CARD
    with col1:

        components.html(
            f"""
            <div style="
                padding:20px;
                border-radius:22px;
                background:white;
                box-shadow:0px 8px 25px rgba(124,58,237,0.15);
                height:220px;
                font-family:sans-serif;
            ">

                <h3 style="color:#7c3aed;">
                🧠 CNN
                </h3>

                <p><b>Prediction:</b> {cnn_pred}</p>

                <p><b>Confidence:</b>
                {cnn_conf*100:.2f}%</p>

            </div>
            """,
            height=240
        )

    # Transformer Card
    with col2:

        components.html(
            f"""
            <div style="
                padding:20px;
                border-radius:22px;
                background:white;
                box-shadow:0px 8px 25px rgba(124,58,237,0.15);
                height:220px;
                font-family:sans-serif;
            ">

                <h3 style="color:#9333ea;">
                🤖 Transformer
                </h3>

                <p><b>Prediction:</b> {vit_pred}</p>

                <p><b>Confidence:</b>
                {vit_conf*100:.2f}%</p>

            </div>
            """,
            height=240
        )

    # Hybrid Card
    with col3:

        components.html(
            f"""
            <div style="
                padding:20px;
                border-radius:22px;
                background:white;
                box-shadow:0px 8px 25px rgba(124,58,237,0.15);
                height:220px;
                font-family:sans-serif;
            ">

                <h3 style="color:#db2777;">
                🌟 Hybrid AI
                </h3>

                <p><b>Prediction:</b> {hybrid_pred}</p>

                <p><b>Confidence:</b>
                {hybrid_conf*100:.2f}%</p>

            </div>
            """,
            height=240
        )

    # ================= COMPARISON TABLE =================
    comparison_df = pd.DataFrame({

        "Model":[
            "CNN",
            "Transformer",
            "Hybrid"
        ],

        "Prediction":[
            cnn_pred,
            vit_pred,
            hybrid_pred
        ],

        "Confidence (%)":[
            round(cnn_conf*100,2),
            round(vit_conf*100,2),
            round(hybrid_conf*100,2)
        ]
    })

    st.markdown("## 📊 Comparison Table")

    st.markdown(
        styled_table(comparison_df),
        unsafe_allow_html=True
    )

    # ================= DIFFERENCE TABLE =================
    diff_df = pd.DataFrame({

        "Feature":[
            "Learning Type",
            "Focus",
            "Strength",
            "Weakness",
            "Prediction",
            "Confidence"
        ],

        "CNN":[
            "Deep Learning",
            "Local Features",
            "Fast & Efficient",
            "Misses Global Context",
            cnn_pred,
            f"{cnn_conf*100:.2f}%"
        ],

        "Transformer":[
            "Attention Based",
            "Whole Image",
            "Better Context Understanding",
            "Needs More Data",
            vit_pred,
            f"{vit_conf*100:.2f}%"
        ],

        "Hybrid AI":[
            "CNN + Transformer",
            "Local + Global",
            "Best Combined Accuracy",
            "Slightly Slower",
            hybrid_pred,
            f"{hybrid_conf*100:.2f}%"
        ]
    })

    st.markdown("## 🧠 CNN vs Transformer vs Hybrid")

    st.markdown(
        styled_table(diff_df),
        unsafe_allow_html=True
    )

    # ================= PROBABILITY TABLES =================
    cnn_df = pd.DataFrame({

        "Flower": class_names,

        "Probability (%)":[
            round(float(p)*100,2)
            for p in cnn_probs
        ]
    }).sort_values(
        by="Probability (%)",
        ascending=False
    )

    vit_df = pd.DataFrame({

        "Flower": class_names,

        "Probability (%)":[
            round(float(p)*100,2)
            for p in vit_probs
        ]
    }).sort_values(
        by="Probability (%)",
        ascending=False
    )

    hybrid_df = pd.DataFrame({

        "Flower": class_names,

        "Probability (%)":[
            round(float(p)*100,2)
            for p in hybrid_probs
        ]
    }).sort_values(
        by="Probability (%)",
        ascending=False
    )

    # ================= BAR CHART =================
    st.markdown("## 📈 Probability Chart")

    chart_df = pd.DataFrame({

        "CNN": cnn_df.head(5)["Probability (%)"].values,

        "Transformer": vit_df.head(5)["Probability (%)"].values,

        "Hybrid": hybrid_df.head(5)["Probability (%)"].values

    },

    index=cnn_df.head(5)["Flower"].values
    )

    st.bar_chart(chart_df)

    # ================= TOP TABLES =================
    st.markdown("## 📊 CNN Top Predictions")

    st.markdown(
        styled_table(cnn_df.head(5)),
        unsafe_allow_html=True
    )

    st.markdown("## 📊 Transformer Top Predictions")

    st.markdown(
        styled_table(vit_df.head(5)),
        unsafe_allow_html=True
    )

    st.markdown("## 📊 Hybrid Top Predictions")

    st.markdown(
        styled_table(hybrid_df.head(5)),
        unsafe_allow_html=True
    )

    # ================= AGREEMENT =================
    if cnn_pred == vit_pred == hybrid_pred:
        st.success("✅ All models agree")

    else:
        st.warning("⚠️ Models produced different predictions")

    # ================= FLOWER INFO =================
    final_flower = hybrid_pred.lower()

    if final_flower in flower_info:

        info = flower_info[final_flower]

        components.html(
            f"""
            <div style="
                padding:25px;
                border-radius:20px;
                background:white;
                box-shadow:0px 8px 25px rgba(124,58,237,0.12);
                margin-top:20px;
                font-family:sans-serif;
            ">

                <h2 style="color:#7c3aed;">
                🌸 Flower Information
                </h2>

                <p><b>Name:</b> {info['name']}</p>

                <p><b>Scientific Name:</b>
                {info['scientific']}</p>

                <p><b>Meaning:</b>
                {info['meaning']}</p>

                <p><b>Uses:</b>
                {info['uses']}</p>

                <p><b>Fun Fact:</b>
                {info['fun_fact']}</p>

            </div>
            """,
            height=300
        )

    # ================= DOWNLOAD =================
    st.download_button(
        "📥 Download Comparison CSV",
        comparison_df.to_csv(index=False),
        "comparison.csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "🌸 Premium Flower AI | CNN + Transformer + Hybrid AI"
)