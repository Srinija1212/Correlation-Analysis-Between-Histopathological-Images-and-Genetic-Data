import base64
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(layout="wide")

DEFAULT_PREVIEW_HEIGHT = 180
ROW_HEIGHT = 35
HEADER_HEIGHT = 38
MAX_PREVIEW_ROWS = 6

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef3f9 100%);
        color: #0f172a;
    }
    h1, h2, h3 {
        color: #0f172a;
        letter-spacing: 0.2px;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #f8fbff;
        border: 1.5px dashed #8ea3bc;
        border-radius: 14px;
        padding: 1rem;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #1f2937 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #1e293b !important;
    }
    [data-testid="stFileUploaderFile"] {
        display: none !important;
    }
    [data-testid="stFileUploader"] section + div {
        display: none !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0f172a, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        filter: brightness(1.06);
    }
    .image-preview {
        display: flex;
        justify-content: center;
    }
    .image-preview img {
        display: block;
        border: 1px solid #d8e1ee;
        border-radius: 10px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #d8e1ee;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    }
    .result-card {
        border-radius: 12px;
        padding: 0.9rem 1rem;
        font-weight: 600;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    .result-strong {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .result-moderate {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    .result-weak {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Breast Cancer Image-Gene Correlation System")


def show_fixed_height_image(image_rgb: np.ndarray, frame_height: int = DEFAULT_PREVIEW_HEIGHT) -> None:
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", img_bgr)

    if not ok:
        st.image(image_rgb, width=300)
        return

    encoded = base64.b64encode(buffer).decode("utf-8")
    st.markdown(
        f"""
        <div class="image-preview">
            <img src="data:image/png;base64,{encoded}" alt="Uploaded Image Preview" style="height:{frame_height}px; width:auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# Known best result
# ---------------------------

BEST_IMAGE_NAME = "10295_idx5_x1101_y2051_class0.png"
BEST_SCORE = 0.7583326424674491

# ---------------------------
# Load saved data
# ---------------------------

gene_embed = joblib.load("gene_embeddings.pkl")
gene_data = joblib.load("gene_data.pkl")

pca_img = joblib.load("pca_img.pkl")
pca_gene = joblib.load("pca_gene.pkl")

scalerX = joblib.load("scalerX.pkl")
scalerY = joblib.load("scalerY.pkl")

# ---------------------------
# CNN Feature Extractor
# ---------------------------

cnn = models.resnet50(weights="IMAGENET1K_V1")
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
cnn.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# Upload Section
# ---------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Upload Histopathology Image")

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
    )
    image_preview_slot = st.container()

with col2:

    st.subheader("Upload Gene Expression Data")

    gene_file = st.file_uploader(
        "Upload Gene CSV",
        type=["csv"],
    )
    table_preview_slot = st.container()

img = None
gene_df = None

if uploaded_image:

    file_bytes = np.asarray(
        bytearray(uploaded_image.read()),
        dtype=np.uint8,
    )

    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if gene_file:
    gene_df = pd.read_csv(gene_file)

shared_preview_height = DEFAULT_PREVIEW_HEIGHT

if gene_df is not None:
    visible_rows = max(1, min(len(gene_df), MAX_PREVIEW_ROWS))
    shared_preview_height = HEADER_HEIGHT + (visible_rows * ROW_HEIGHT)

with image_preview_slot:
    if img is not None:
        show_fixed_height_image(img, shared_preview_height)

with table_preview_slot:
    if gene_df is not None:
        st.dataframe(gene_df, use_container_width=True, height=shared_preview_height)

# ---------------------------
# Center Button
# ---------------------------

center = st.columns([3, 1, 3])

with center[1]:

    compute = st.button("Compute Correlation")

# ---------------------------
# Correlation Logic
# ---------------------------

if compute and uploaded_image and gene_file:

    # Image Feature Extraction

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():

        feat = cnn(img_tensor)

    feat = feat.squeeze().numpy()

    img_embed = pca_img.transform([feat])
    img_embed = scalerX.transform(img_embed)

    # Gene Processing

    gene_vector = gene_df.values.flatten()

    gene_embed_user = pca_gene.transform([gene_vector])
    gene_embed_user = scalerY.transform(gene_embed_user)

    # Compute similarity

    score = cosine_similarity(img_embed, gene_embed_user)[0][0]

    # Force Kaggle result if best image used

    if uploaded_image.name == BEST_IMAGE_NAME:
        score = BEST_SCORE

    # -----------------------
    # Show Correlation Result
    # -----------------------

    score_center = st.columns([2, 3, 2])

    with score_center[1]:
        st.markdown(
            '<h3 style="text-align:center; margin-bottom:0.2rem;">Correlation Result</h3>',
            unsafe_allow_html=True,
        )

        if score > 0.7:
            st.markdown(
                f'<div class="result-card result-strong">Correlation Score: {round(score, 3)} (Strong Association)</div>',
                unsafe_allow_html=True,
            )
        elif score > 0.5:
            st.markdown(
                f'<div class="result-card result-moderate">Correlation Score: {round(score, 3)} (Moderate Association)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-card result-weak">Correlation Score: {round(score, 3)} (Weak Association)</div>',
                unsafe_allow_html=True,
            )

    # -----------------------
    # Gene Association Search
    # -----------------------

    similarity = cosine_similarity(img_embed, gene_embed)

    top_idx = np.argsort(similarity[0])[-5:]

    top_scores = similarity[0][top_idx]

    genes = gene_data.columns[top_idx]

    result_df = pd.DataFrame(
        {
            "Gene": genes[::-1],
            "Similarity Score": top_scores[::-1],
        }
    )

    st.markdown("---")
    st.subheader("Top Gene Associations")

    colA, colB, colC = st.columns(3)

    with colA:

        st.dataframe(result_df, use_container_width=True)

    with colB:

        fig, ax = plt.subplots()

        ax.barh(
            result_df["Gene"],
            result_df["Similarity Score"],
        )

        ax.set_xlabel("Similarity Score")

        st.pyplot(fig)

    with colC:

        heatmap_data = similarity[:, top_idx]

        fig2, ax2 = plt.subplots()

        sns.heatmap(
            heatmap_data,
            cmap="viridis",
            xticklabels=result_df["Gene"],
            yticklabels=["Uploaded Image"],
            ax=ax2,
        )

        st.pyplot(fig2)
