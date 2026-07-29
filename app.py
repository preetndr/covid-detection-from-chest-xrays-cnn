import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="COVID-19 X-Ray Classifier", page_icon="🫁", layout="centered"
)

st.markdown(
    """
    <style>

    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700,800&f[]=satoshi@400,500,700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"], .stMarkdown, p, li {
        font-family: 'Satoshi', sans-serif !important;
        color: #F0EDCC !important; 
    }
    
    /* Native Streamlit Dark Background */
    .stApp {
        background-color: #0e1117 !important;
    }

    h1, h2, h3, .gradient-text, .section-header {
        font-family: 'Clash Display', sans-serif !important;
    }

    /* Keep hidden accessibility labels hidden */
    .st-visually-hidden, [class*="visually-hidden"] {
        display: none !important;
    }

    /* --- PREMIUM CANVA-STYLE DROPZONE BASE STYLES --- */
    /* (The Javascript at the bottom handles the inner content to prevent glitches) */
    
    [data-testid="stFileUploaderDropzone"], 
    [data-testid="stFileUploadDropzone"], 
    [data-testid="stFileUploader"] section {
        background-color: rgba(2, 52, 63, 0.2) !important; 
        border: none !important;
        border-radius: 16px !important;
        min-height: 240px !important;
        position: relative !important;
        display: block !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.3s ease !important;
        overflow: hidden !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover, 
    [data-testid="stFileUploadDropzone"]:hover, 
    [data-testid="stFileUploader"] section:hover {
        background-color: rgba(2, 52, 63, 0.35) !important;
    }

    /* Inner Dashed Border */
    [data-testid="stFileUploaderDropzone"]::before, 
    [data-testid="stFileUploadDropzone"]::before, 
    [data-testid="stFileUploader"] section::before {
        content: "";
        position: absolute;
        inset: 14px;
        border: 2px dashed rgba(240, 237, 204, 0.25);
        border-radius: 10px;
        pointer-events: none;
        transition: border-color 0.3s ease;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover::before, 
    [data-testid="stFileUploadDropzone"]:hover::before, 
    [data-testid="stFileUploader"] section:hover::before {
        border-color: rgba(56, 189, 248, 0.4);
    }
    /* --- END CANVA DROPZONE BASE --- */

    /* Section Headers */
    .section-header {
        color: #02343F;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 62px;
        margin-bottom: 25px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(240, 237, 204, 0.1);
        background: none;
        display: block;
    }

    /* Analyze Button Styling */
    .element-container:has(.stButton) {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        margin-top: 18px !important;
    }
    div.stButton {
        width: auto !important;
        display: flex !important;
        justify-content: center !important;
    }
    div.stButton > button {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 13px 30px !important;
        border-radius: 8px !important;
        background: rgba(2, 52, 63, 0.15) !important;
        border: 1px solid rgba(2, 52, 63, 0.4) !important;
        color: #F0EDCC !important;
        backdrop-filter: blur(10px) !important;
        overflow: hidden !important;
        transition: transform 0.45s cubic-bezier(0.16, 1, 0.3, 1), background 0.45s ease, border-color 0.45s ease, box-shadow 0.45s ease !important;
        box-shadow: 0 0 0 rgba(2, 52, 63, 0), 0 8px 30px rgba(0,0,0,0.18);
        
        /* Clean text layout */
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 14px !important; 
        font-weight: 800 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        white-space: nowrap !important;
    }

    div.stButton > button::before {
        content: "";
        position: absolute;
        inset: -40%;
        background: radial-gradient(circle at center, rgba(2, 52, 63, 0.4) 0%, rgba(2, 52, 63, 0.1) 30%, transparent 70%);
        opacity: 0;
        transform: translateX(-30%) translateY(10%) scale(0.8);
        transition: opacity 0.6s ease, transform 0.8s cubic-bezier(0.16,1,0.3,1);
        pointer-events: none;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.015);
        background: rgba(2, 52, 63, 0.3) !important;
        border-color: #02343F !important;
        box-shadow: 0 0 40px rgba(2, 52, 63, 0.3), 0 12px 40px rgba(0,0,0,0.24);
    }
    div.stButton > button:hover::before {
        opacity: 1;
        transform: translateX(15%) translateY(-10%) scale(1.15);
    }
    div.stButton > button:active {
        transform: translateY(0px) scale(0.985);
    }

    /* Animations & Background */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes textShine {
        0% { background-position: 0% center; }
        100% { background-position: 100% center; }
    }
    .gradient-text {
        background: linear-gradient(120deg, #02343F 30%, #035263 50%, #02343F 70%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textShine 4s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
    }

    .block-container {
        margin-top: -64px !important;
    }

    /* Background dots colored in subtle Teal */
    body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.05;
        z-index: 0;
        background-image: radial-gradient(#02343F 1px, transparent 1px);
        background-size: 8px 8px;
    }

    .main .block-container {
        animation: pageReveal 900ms cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes pageReveal {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-reveal {
        animation: resultReveal 650ms cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes resultReveal {
        from { opacity: 0; transform: translateY(14px); filter: blur(6px); }
        to { opacity: 1; transform: translateY(0); filter: blur(0px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_VGG.h5")
    return tf.keras.models.load_model(model_path)


model_loaded = False
try:
    model = load_model()
    class_labels = ["COVID", "Normal", "Viral Pneumonia"]
    model_loaded = True
except FileNotFoundError:
    load_error = "model_VGG.h5 not found. Make sure it is in the same folder as app.py."
except Exception as e:
    load_error = str(e)


st.markdown(
    """
    <h1 style='font-size: 60px; margin-bottom: 0; line-height: 1.1; text-align: center;'>
        COVID-19 X-Ray
        <span class='gradient-text' style='display: block; font-size: 68px; font-weight: 800;'>Classifier</span>
    </h1>
    <p style='color: #888; font-size: 16px; margin-top: 20px; text-align: center; font-weight: 500;'>
        Powered by VGG16 Deep Learning Architecture
    </p>
    <div style='height: 34px;'></div>
    """,
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(f"⚠️ Could not load model — {load_error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload X-Ray Scan (JPG, PNG)",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed",
)

st.markdown("<div style='height: 34px;'></div>", unsafe_allow_html=True)

if uploaded_file is not None:

    # Show preview of uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_container_width=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    if st.button("Analyze Image"):
        try:
            # 1. Preprocess the image to match the VGG16 input (224, 224, 3)
            if image.mode != "RGB":
                image = image.convert("RGB")

            img = image.resize((224, 224))
            img_array = np.array(img)

            # The model expects a batch, so we add an extra dimension: (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # Scale pixels exactly like we did in the datagen (1./255)
            img_array = img_array / 255.0

            # 2. Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = float(np.max(predictions)) * 100

            final_diagnosis = class_labels[predicted_class_index]

            st.markdown(
                """
                <div class="result-reveal" style="text-align:center; margin-top:38px; padding:42px 20px 12px;">
                    <p style="color:#777; font-size:11px; font-weight:700; letter-spacing:0.22em; text-transform:uppercase; margin:0 0 18px;">
                        Diagnostic Result
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<h1 style='text-align:center; font-size:68px; margin:0; font-family:Clash Display, sans-serif; line-height:1;' class='gradient-text'>{final_diagnosis}</h1>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='width:46px; height:1px; background:rgba(240, 237, 204, 0.22); margin:28px auto 0;'></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="result-reveal" style="background: rgba(2, 52, 63, 0.15); border: 1px solid rgba(2, 52, 63, 0.4); border-left: 3px solid #02343F; border-radius: 8px; padding: 1.2rem; margin-top: 40px; backdrop-filter: blur(10px);">
                    <p style="color: #F0EDCC; font-size: 0.9rem; margin: 0; line-height: 1.5; font-family: 'Plus Jakarta Sans', sans-serif;">
                        <span style="color: #02343F; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.75rem;">Analysis Complete</span><br><br>
                        The neural network has classified this scan as <strong style="color: #02343F;">{final_diagnosis}</strong> with a confidence score of <strong>{confidence_score:.2f}%</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Auto-scroll to results
            st.components.v1.html(
                """
                <script>
                    const result = window.parent.document.querySelector('.result-reveal');
                    if (result) {
                        result.scrollIntoView({ behavior: 'smooth' });
                    }
                </script>
                """,
                height=0,
            )

        except Exception as e:
            st.error(f"Computation Error: {e}")

st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

st.components.v1.html(
    """
    <script>
    const doc = window.parent.document;

    // --- BUTTON HOVER ANIMATIONS ---
    function centerButton() {
        doc.querySelectorAll('.stButton').forEach(el => {
            el.style.setProperty('display', 'flex', 'important');
            el.style.setProperty('justify-content', 'center', 'important');
            let parent = el.parentElement;
            while (parent) {
                parent.style.setProperty('display', 'flex', 'important');
                parent.style.setProperty('justify-content', 'center', 'important');
                if (parent.classList.contains('block-container')) break;
                parent = parent.parentElement;
            }
        });
    }

    centerButton();

    // --- BULLETPROOF CANVA UPLOADER OVERRIDE ---
    // This script finds Streamlit's native uploader box, hides its glitchy text, 
    // and paints the beautiful Canva layout over it, making the entire box a hidden clickable button.
    function upgradeUploader() {
        const dropzones = doc.querySelectorAll('[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"], [data-testid="stFileUploader"] section');
        
        dropzones.forEach(dropzone => {
            if (!dropzone.classList.contains('canva-upgraded')) {
                dropzone.classList.add('canva-upgraded');
                
                // 1. Hide default Streamlit text and junk
                Array.from(dropzone.children).forEach(child => {
                    if (child.tagName !== 'BUTTON') {
                        child.style.display = 'none';
                    }
                });
                
                // 2. Build the gorgeous Canva-style visual layer
                const overlay = doc.createElement('div');
                overlay.innerHTML = `
                    <div style="pointer-events: none; display: flex; flex-direction: column; align-items: center; justify-content: center; position: absolute; inset: 0; z-index: 5;">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#38bdf8" width="56" height="56" style="margin-bottom: 12px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.25));">
                            <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
                        </svg>
                        <p style="margin: 0; color: #F0EDCC; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 16px; font-weight: 600; letter-spacing: 0.5px;">
                            Drop your image here, or <span style="color: #38bdf8; text-decoration: underline;">browse</span>
                        </p>
                        <p style="margin: 8px 0 0; color: rgba(240, 237, 204, 0.4); font-size: 12px; font-weight: 500;">
                            Supports: JPG, JPEG, PNG
                        </p>
                    </div>
                `;
                dropzone.appendChild(overlay);
                
                // 3. Find the browse button, make it 100% invisible, and stretch it over EVERYTHING
                let btn = dropzone.querySelector('button');
                if (!btn) {
                    const parent = dropzone.closest('[data-testid="stFileUploader"]');
                    if (parent) btn = parent.querySelector('button');
                }
                
                if (btn) {
                    btn.style.setProperty('position', 'absolute', 'important');
                    btn.style.setProperty('top', '0', 'important');
                    btn.style.setProperty('left', '0', 'important');
                    btn.style.setProperty('width', '100%', 'important');
                    btn.style.setProperty('height', '100%', 'important');
                    btn.style.setProperty('opacity', '0', 'important'); // Completely invisible!
                    btn.style.setProperty('z-index', '999', 'important'); // Sitting right on top
                    btn.style.setProperty('cursor', 'pointer', 'important');
                }
            }
        });
    }

    upgradeUploader();

    // Re-run scripts if Streamlit reloads the DOM
    const observer = new MutationObserver(() => {
        centerButton();
        upgradeUploader();
    });
    observer.observe(doc.body, { childList: true, subtree: true });

    </script>
    """,
    height=0,
    width=0,
)
