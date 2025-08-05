# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import re
# import fitz  # PyMuPDF
# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # 🔐 Set up Gemini
# if GEMINI_API_KEY:
#     genai.configure(api_key=GEMINI_API_KEY)
#     gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
# else:
#     gemini_model = None

# # Load Sentence-BERT
# @st.cache_resource
# def load_model():
#     return SentenceTransformer('all-MiniLM-L6-v2')

# model = load_model()

# # 📄 Extract text from PDF
# def extract_text_from_pdf(uploaded_file):
#     with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
#         text = ""
#         for page in doc:
#             text += page.get_text()
#     return text

# # Clean text
# def clean_text(text):
#     return re.sub(r'\s+', ' ', text).strip().lower()

# # Compute cosine similarity
# def compute_similarity(resume, jd):
#     embeddings = model.encode([resume, jd])
#     score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#     return round(score * 100, 2)

# # Keyword extraction
# def extract_keywords(text):
#     words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
#     return set(words)

# # Gemini Suggestions
# def get_gemini_suggestions(resume, jd):
#     prompt = f"""
# You are a resume reviewer AI. Given the following resume and job description,
# provide specific suggestions to improve the resume so it better matches the job.

# Resume:
# \"\"\"
# {resume}
# \"\"\"

# Job Description:
# \"\"\"
# {jd}
# \"\"\"

# Return 3–5 concise, actionable improvement suggestions. 

# Rate how well this resume matches the job description from 1 to 10.
# Then give a one-paragraph reason. Be honest but constructive.

# Resume:
# \"\"\"
# {resume}
# \"\"\"

# Job Description:
# \"\"\"
# {jd}
# \"\"\"

# Return format:
# Score: x/10
# Reason: ...
#     """
#     if gemini_model:
#         try:
#             response = gemini_model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             return f"Error from Gemini: {e}"
#     return "Gemini API key not found or not configured."

# # 🚀 Streamlit UI
# st.set_page_config(page_title="AI Resume Evaluator", layout="wide")
# st.title("📄 AI-Powered Resume Evaluator (w/ Gemini)")
# st.markdown("Upload your **resume** and paste the **job description** to evaluate the match.")

# col1, col2 = st.columns(2)

# # PDF Upload
# with col1:
#     uploaded_file = st.file_uploader("📁 Upload Resume PDF", type=["pdf"])
#     if uploaded_file:
#         resume_text = extract_text_from_pdf(uploaded_file)
#         st.success("✅ Resume text extracted from PDF!")
#     else:
#         resume_text = st.text_area("✍️ Or Paste Resume Text", height=250)

# # Job Description
# with col2:
#     jd_text = st.text_area("📌 Paste Job Description", height=400)

# # Evaluate Button
# if st.button("🔍 Evaluate"):
#     if resume_text and jd_text:
#         resume_clean = clean_text(resume_text)
#         jd_clean = clean_text(jd_text)

#         # Similarity
#         score = compute_similarity(resume_clean, jd_clean)
        
#         # Keyword Analysis
#         resume_keywords = extract_keywords(resume_clean)
#         jd_keywords = extract_keywords(jd_clean)
#         missing_keywords = jd_keywords - resume_keywords

        
#         # Gemini Suggestions
#         st.subheader("💡 Suggestions to Improve Resume (Gemini)")
#         with st.spinner("Thinking..."):
#             suggestions = get_gemini_suggestions(resume_text, jd_text)
#         st.markdown(suggestions)
#     else:
#         st.error("Please upload or paste both resume and job description.")

import streamlit as st

# Set full width layout
st.set_page_config(page_title="AI Resume Evaluator", layout="wide")

# Hide default Streamlit elements (footer, menu)
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 2rem;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ---- Title ----
st.markdown("<h1 style='text-align: center;'>AI-Powered Resume Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Match your resume with job descriptions and get intelligent feedback powered by Gemini LLM</p>", unsafe_allow_html=True)

# ---- Buttons as Navigation ----
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### 🚀 Get Started")
    st.page_link("pages/upload_resume.py", label="Upload Resume & Job Description", icon="📄")
    st.page_link("pages/results.py", label="View Evaluation Results", icon="📈")

# ---- Optional Note or Footer ----
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9rem; color: grey;'>Built with ❤️ using Streamlit and Gemini 2.5 Pro</p>",
    unsafe_allow_html=True
)

