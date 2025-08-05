import streamlit as st
import fitz  # PyMuPDF

# -------------------- Config & Style --------------------
st.set_page_config(page_title="Upload Resume & JD", layout="wide")

# Hide Streamlit branding
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 2rem;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# -------------------- Session State Init --------------------
if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

# -------------------- Page Title --------------------
st.markdown("<h2 style='text-align: center;'>📤 Upload Resume & Job Description</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your resume (PDF) and paste the job description to begin evaluation</p>", unsafe_allow_html=True)

# -------------------- Upload and JD Form --------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # --- Upload Resume ---
    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
    if uploaded_file:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        st.session_state.resume_text = text
        st.success("✅ Resume uploaded and text extracted!")

    # --- Paste Job Description ---
    jd_text = st.text_area("📝 Paste Job Description", value=st.session_state.jd_text, height=300)
    st.session_state.jd_text = jd_text

    # --- Submit Button ---
    if st.button("🚀 Submit for Evaluation"):
        if st.session_state.resume_text.strip() and st.session_state.jd_text.strip():
            st.session_state["submitted"] = True
            st.success("✅ Analysis submitted. Redirecting to Results page...")
            st.rerun()  # Use rerun for immediate UI refresh
        else:
            st.error("⚠️ Please upload a resume and paste a job description before submitting.")

# -------------------- Auto Redirect if Already Submitted --------------------
if st.session_state.submitted:
    st.switch_page("pages/results.py")
