import os
import io
import base64
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="COVID-19 X-Ray Classifier", page_icon="🫁", layout="centered"
)

# Initialize a session state key to forcefully reset the file uploader and handle scrolling
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

st.markdown(
    """
    <style>
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700,800&f[]=satoshi@400,500,700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"], .stMarkdown, p, li {
        font-family: 'Satoshi', sans-serif !important;
    }
    
    /* Native Streamlit Dark Background */
    .stApp {
        background-color: #0e1117 !important;
    }

    h1, h2, h3, .gradient-text, .section-header {
        font-family: 'Clash Display', sans-serif !important;
    }

    /* Target the uploader container to center things */
    [data-testid="stFileUploader"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
    }

    /* Button Styling */
    div.stButton > button {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        padding: 14px 20px !important;
        border-radius: 8px !important;
        background: rgba(9, 132, 158, 0.15) !important;
        border: 1px solid rgba(9, 132, 158, 0.4) !important;
        color: #F0EDCC !important;
        backdrop-filter: blur(10px) !important;
        overflow: hidden !important;
        transition: transform 0.45s cubic-bezier(0.16, 1, 0.3, 1), background 0.45s ease, border-color 0.45s ease, box-shadow 0.45s ease !important;
        box-shadow: 0 0 0 rgba(9, 132, 158, 0), 0 8px 30px rgba(0,0,0,0.18);
        
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 10px !important; 
        font-weight: 800 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        white-space: nowrap !important;
    }

    div.stButton > button::before {
        content: "";
        position: absolute;
        inset: -40%;
        background: radial-gradient(circle at center, rgba(21, 181, 214, 0.4) 0%, rgba(21, 181, 214, 0.1) 30%, transparent 70%);
        opacity: 0;
        transform: translateX(-30%) translateY(10%) scale(0.8);
        transition: opacity 0.6s ease, transform 0.8s cubic-bezier(0.16,1,0.3,1);
        pointer-events: none;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.015);
        background: rgba(21, 181, 214, 0.3) !important;
        border-color: #15B5D6 !important;
        box-shadow: 0 0 40px rgba(21, 181, 214, 0.3), 0 12px 40px rgba(0,0,0,0.24);
    }
    div.stButton > button:hover::before {
        opacity: 1;
        transform: translateX(15%) translateY(-10%) scale(1.15);
    }
    div.stButton > button:active {
        transform: translateY(0px) scale(0.985);
    }

    @keyframes textShine {
        0% { background-position: 0% center; }
        100% { background-position: 100% center; }
    }
    .gradient-text {
        background: linear-gradient(120deg, #09849E 30%, #15B5D6 50%, #09849E 70%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textShine 4s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
    }

    .block-container {
        margin-top: -64px !important;
    }

    body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.05;
        z-index: 0;
        background-image: radial-gradient(#09849E 1px, transparent 1px);
        background-size: 8px 8px;
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
    <style>
    /* 1. Create a Sticky Header with a Frosted Glass Fog Effect */
    .sticky-header-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 99999;
        padding-top: 65px; 
        padding-bottom: 40px; 
        pointer-events: none;
        
        /* Glassmorphism Fog Effect */
        background: linear-gradient(to bottom, rgba(14, 17, 23, 0.95) 45%, rgba(14, 17, 23, 0) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        
        /* Smoothly fades out the blur so there are no harsh bottom edges */
        -webkit-mask-image: linear-gradient(to bottom, black 65%, transparent 100%);
        mask-image: linear-gradient(to bottom, black 65%, transparent 100%);
    }
    
    /* 2. Hide Streamlit's default top-right menu to prevent overlap */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    </style>

    <div class="sticky-header-container">
        <h1 style='color: #F0EDCC; font-size: 60px; margin-top: 0; margin-bottom: 0; line-height: 1.1; text-align: center; pointer-events: auto;'>
            COVID-19 X-Ray
            <span class='gradient-text' style='display: block; font-size: 68px; font-weight: 800;'>Classifier</span>
        </h1>
    </div>
    
    <!-- EXACT physical invisible spacer (Set precisely to 287px) -->
    <div class="top-reveal" style="height: 287px; width: 100%; pointer-events: none;"></div>
    
    <!-- Subtitle in normal flow so it scrolls away naturally -->
    <p style='color: #888; font-size: 16px; margin-top: 0px; margin-bottom: 25px; text-align: center; font-weight: 500;'>
        Powered by VGG16 Deep Learning Architecture
    </p>
    """,
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(f"⚠️ Could not load model — {load_error}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload X-Ray Scan",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed",
    key=f"uploader_{st.session_state.file_uploader_key}",
)

if uploaded_file is None:
    # Execute absolute scroll to top if the remove button was just clicked
    if st.session_state.get("scroll_to_top", False):
        st.components.v1.html(
            """
            <script>
                // Target Streamlit's internal scrolling container directly and force it to 0
                const stMain = window.parent.document.querySelector('.stMain') || window.parent.document.querySelector('.main');
                if (stMain) {
                    stMain.scrollTo({ top: 0, behavior: 'smooth' });
                } else {
                    window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                }
            </script>
            """,
            height=0,
        )
        st.session_state.scroll_to_top = False

    st.markdown(
        """
        <style>
        /* --- PREMIUM CANVA-STYLE DROPZONE BASE STYLES --- */
        [data-testid="stFileUploaderDropzone"], 
        [data-testid="stFileUploadDropzone"], 
        [data-testid="stFileUploader"] section {
            background-color: rgba(9, 132, 158, 0.15) !important; 
            border: none !important;
            border-radius: 16px !important;
            min-height: 240px !important;
            width: 100% !important;
            position: relative !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important;
            transition: all 0.3s ease !important;
            overflow: hidden !important;
            margin: 20px auto !important;
        }
        
        [data-testid="stFileUploaderDropzone"]:hover, 
        [data-testid="stFileUploadDropzone"]:hover, 
        [data-testid="stFileUploader"] section:hover {
            background-color: rgba(9, 132, 158, 0.3) !important;
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
            border-color: rgba(21, 181, 214, 0.6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.components.v1.html(
        """
        <script>
        function upgradeUploader() {
            const doc = window.parent.document;
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
                    
                    // 2. Build the Canva-style visual layer using our Light Teal theme
                    const overlay = doc.createElement('div');
                    overlay.innerHTML = `
                        <div style="pointer-events: none; display: flex; flex-direction: column; align-items: center; justify-content: center; position: absolute; inset: 0; z-index: 5;">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#15B5D6" width="56" height="56" style="margin-bottom: 12px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.25));">
                                <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
                            </svg>
                            <p style="margin: 0; color: #F0EDCC; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 16px; font-weight: 600; letter-spacing: 0.5px;">
                                Drop your image here, or <span style="color: #15B5D6; text-decoration: underline;">browse</span>
                            </p>
                            <p style="margin: 8px 0 0; color: rgba(240, 237, 204, 0.4); font-size: 12px; font-weight: 500;">
                                Supports: JPG, JPEG, PNG
                            </p>
                        </div>
                    `;
                    dropzone.appendChild(overlay);
                    
                    // 3. Stretch the invisible HTML file input over the whole box
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
                        btn.style.setProperty('opacity', '0', 'important'); 
                        btn.style.setProperty('z-index', '999', 'important'); 
                        btn.style.setProperty('cursor', 'pointer', 'important');
                    }
                }
            });
        }

        upgradeUploader();

        // Setup observer to handle Streamlit DOM refreshes dynamically
        const observer = new MutationObserver(() => {
            upgradeUploader();
        });
        observer.observe(window.parent.document.body, { childList: true, subtree: true });
        </script>
        """,
        height=0,
        width=0,
    )

else:
    image = Image.open(uploaded_file)
    bg_image = image.copy()
    bg_image.thumbnail((800, 800))
    if bg_image.mode != "RGB":
        bg_image = bg_image.convert("RGB")

    buffered = io.BytesIO()
    bg_image.save(buffered, format="JPEG", quality=85)
    b64_encoded = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <style>
        /* 1. Completely destroy the native Dropzone button so it stops overlapping */
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploadDropzone"] {{
            display: none !important;
        }}

        /* 2. Paint the uploaded image onto the main wrapper */
        div[data-testid="stFileUploader"] {{
            background-image: url("data:image/jpeg;base64,{b64_encoded}") !important;
            background-size: contain !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            height: 400px !important;
            min-height: 400px !important;
            width: 100% !important;
            position: relative !important;
            margin-top: 1rem !important;
            background-color: transparent !important;
            border: none !important;
        }}

        /* 3. Mathematically annihilate ALL file text, icons, AND the native close button */
        div[data-testid="stUploadedFile"] p,
        div[data-testid="stUploadedFile"] small,
        div[data-testid="stUploadedFile"] svg,
        div[data-testid="stUploadedFile"] button {{
            display: none !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    col_spacer1, col1, col2, col_spacer2 = st.columns([1, 1.5, 1.5, 1])

    with col1:
        analyze_clicked = st.button("Analyze Image", use_container_width=True)

    with col2:
        # Programmatically resets the Uploader's session state key and flags scroll
        if st.button("Remove Image", use_container_width=True):
            st.session_state.scroll_to_top = True
            st.session_state.file_uploader_key += 1
            st.rerun()

    # We use position: absolute to place the target exactly 55px below the buttons
    # Because height is 0px, it does NOT physically push the layout around or break Streamlit's flexbox!
    st.markdown(
        """
        <div style='position: relative; width: 100%; height: 0px;'>
            <div class='upload-reveal' style='position: absolute; top: 55px; left: 0; width: 1px; height: 1px;'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Only run the image auto-scroll if we haven't clicked Analyze yet!
    # This stops the screen from jumping around when the result loads.
    if not analyze_clicked:
        st.components.v1.html(
            """
            <script>
                const uploadTarget = window.parent.document.querySelector('.upload-reveal');
                if (uploadTarget) {
                    // block: 'end' aligns the bottom of our 75px ghost target with the bottom of the viewport
                    uploadTarget.scrollIntoView({ behavior: 'smooth', block: 'end' });
                }
            </script>
            """,
            height=0,
        )

    if analyze_clicked:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = float(np.max(predictions)) * 100
            final_diagnosis = class_labels[predicted_class_index]

            st.markdown(
                """
                <div class="result-reveal" style="text-align:center; margin-top:103px; padding:42px 20px 12px; scroll-margin-top: 280px;">
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
                <div class="result-reveal" style="background: rgba(9, 132, 158, 0.15); border: 1px solid rgba(9, 132, 158, 0.4); border-left: 3px solid #09849E; border-radius: 8px; padding: 1.2rem; margin-top: 40px; backdrop-filter: blur(10px);">
                    <p style="color: #F0EDCC; font-size: 0.9rem; margin: 0; line-height: 1.5; font-family: 'Plus Jakarta Sans', sans-serif; text-align: center;">
                        <span style="color: #15B5D6; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.75rem;">Analysis Complete</span><br><br>
                        The neural network has classified this scan as <strong style="color: #15B5D6;">{final_diagnosis}</strong> with a confidence score of <strong>{confidence_score:.2f}%</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Auto-scroll down to results
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
            st.error(f"Error during analysis: {e}")

st.markdown(
    "<div style='height: 1px; margin-bottom: -5px;'></div>", unsafe_allow_html=True
)
