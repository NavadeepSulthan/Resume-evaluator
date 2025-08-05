import streamlit as st
import os
import re
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter

# ──────────────────────────────────────────────────────────────
# 🔐 Gemini API Setup
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return set(words)

def compute_cosine_similarity(resume, jd):
    embeddings = model.encode([resume, jd])
    return round(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100, 2)

def compute_keyword_coverage(resume_keywords, jd_keywords):
    if not jd_keywords:
        return 0.0
    matched = resume_keywords & jd_keywords
    return round(len(matched) / len(jd_keywords) * 100, 2)

SKILL_SET = {
    'python', 'java', 'c++', 'tensorflow', 'pytorch', 'nlp',
    'machine learning', 'deep learning', 'sql', 'mongodb',
    'fastapi', 'django', 'react', 'aws', 'docker', 'git', 'rest api'
}

def compute_skill_match(resume_text):
    resume_lower = resume_text.lower()
    matched_skills = [skill for skill in SKILL_SET if skill in resume_lower]
    return round(len(matched_skills) / len(SKILL_SET) * 100, 2), matched_skills

def compute_repeated_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    repeated = {word: count for word, count in counts.items() if count > 2}
    return repeated

def plot_pie(title, labels, sizes):
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

def generate_prompts(resume, jd):
    return {
        "Match Summary": f"""Analyze how well the resume matches the job description.""",
        "Pros": f"""List the strengths in the resume based on the job description.""",
        "Cons": f"""List the weaknesses in the resume based on the job description.""",
        "Suggestions": f"""Suggest 3–5 specific improvements for the resume to better match the JD.""",
        "Level Check": f"""Classify the resume's suitability as Entry, Mid, or Senior-level and mention signs of leadership or initiative. and at the end also include the ATS score.""",
        "Grammar Mistakes": f"""Identify grammar and language issues in the resume. Focus only on grammar, spelling, and clarity.""",
    }

# ──────────────────────────────────────────────────────────────
# 🚀 Streamlit Page
st.set_page_config(page_title="Resume Evaluation Results", layout="wide")
st.title("📊 Resume Evaluation Results")

if "resume_text" not in st.session_state or "jd_text" not in st.session_state:
    st.warning("Please upload a resume and job description from Page 1.")
    st.stop()

resume_text = st.session_state.resume_text
jd_text = st.session_state.jd_text
resume_clean = clean_text(resume_text)
jd_clean = clean_text(jd_text)

resume_keywords = extract_keywords(resume_clean)
jd_keywords = extract_keywords(jd_clean)

cos_sim = compute_cosine_similarity(resume_clean, jd_clean)
keyword_score = compute_keyword_coverage(resume_keywords, jd_keywords)
skill_score, matched_skills = compute_skill_match(resume_text)
ats_score = round((cos_sim + keyword_score + skill_score) / 3, 2)
repeated_words = compute_repeated_words(resume_text)

# ─────────────────────────────────────────────
st.subheader("🔢 Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔁 Cosine Similarity", f"{cos_sim} %")
col2.metric("🔑 Keyword Coverage", f"{keyword_score} %")
col3.metric("🛠️ Skill Match Score", f"{skill_score} %")
col4.metric("📌 ATS Score", f"{ats_score} %")

# ─────────────────────────────────────────────
st.subheader("📈 Visual Insights")
col5, col6 = st.columns(2)

with col5:
    st.markdown("**Keyword Coverage**")
    matched = len(resume_keywords & jd_keywords)
    missing = len(jd_keywords - resume_keywords)
    plot_pie("Keyword Coverage", ["Matched", "Missing"], [matched, missing])

with col6:
    st.markdown("**Skill Match**")
    missing_skills = len(SKILL_SET - set(matched_skills))
    plot_pie("Skill Match", ["Matched", "Missing"], [len(matched_skills), missing_skills])

# ─────────────────────────────────────────────
st.subheader("🔁 Repeated Words (3+ times)")
if repeated_words:
    for word, count in repeated_words.items():
        st.write(f"• **{word}** appears **{count}** times")
else:
    st.info("✅ No highly repeated words found.")

# ─────────────────────────────────────────────
st.subheader("🤖 Gemini-Powered Feedback")

prompts = generate_prompts(resume_text, jd_text)

for section, prompt in prompts.items():
    st.markdown(f"### 🔹 {section}")
    with st.spinner(f"Generating {section.lower()}..."):
        feedback = call_gemini(f"""Resume:\n\"\"\"{resume_text}\"\"\"\n\nJD:\n\"\"\"{jd_text}\"\"\"\n\n{prompt}""")
        st.markdown(feedback)
