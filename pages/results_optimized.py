import streamlit as st
import os
import re
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
# ──────────────────────────────────────────────────────────────
# 🔐 Gemini API Setup
start_time = time.time()

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Thread pool for blocking tasks
executor = ThreadPoolExecutor()

# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

# ✅ Extract keywords & skills using Gemini
def extract_keywords_with_gemini_sync(text, context):
    try:
        prompt = f"""
        Extract a list of the most important keywords and skills from the following {context}.
        Focus on technical skills, tools, and domain-specific terms.
        Return them as a comma-separated list without extra commentary.

        Text:
        \"\"\"{text}\"\"\""""
        response = gemini_model.generate_content(prompt)
        keywords = [kw.strip().lower() for kw in response.text.split(",") if kw.strip()]
        return set(keywords)
    except Exception:
        return set()

async def extract_keywords_with_gemini(text, context):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, extract_keywords_with_gemini_sync, text, context)

def compute_cosine_similarity(resume, jd):
    embeddings = model.encode([resume, jd])
    return round(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100, 2)

def compute_keyword_coverage(resume_keywords, jd_keywords):
    if not jd_keywords:
        return 0.0
    matched = resume_keywords & jd_keywords
    return round(len(matched) / len(jd_keywords) * 100, 2)

def compute_skill_match(resume_keywords, jd_keywords):
    if not jd_keywords:
        return 0.0, []
    matched_skills = list(resume_keywords & jd_keywords)
    return round(len(matched_skills) / len(jd_keywords) * 100, 2), matched_skills

def compute_repeated_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    return {word: count for word, count in counts.items() if count > 2}

def plot_pie(title, labels, sizes):
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

def call_gemini_sync(prompt):
    try:
        return gemini_model.generate_content(prompt).text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

async def call_gemini(prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, call_gemini_sync, prompt)

def generate_prompts(resume, jd):
    return {
        "Match Summary": f"""Analyze how well the resume matches the job description.""",
        "Pros": f"""List the strengths in the resume based on the job description.""",
        "Cons": f"""List the weaknesses in the resume based on the job description.""",
        "Suggestions": f"""Suggest 3–5 specific improvements for the resume to better match the JD.""",
        "Level Check": f"""Classify the resume's suitability as Entry, Mid, or Senior-level and include ATS score.""",
        "Grammar Mistakes": f"""Identify grammar and language issues in the resume."""
    }

async def main(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # Stage 1 – Parallel keyword extraction
    resume_keywords, jd_keywords = await asyncio.gather(
        extract_keywords_with_gemini(resume_clean, "resume"),
        extract_keywords_with_gemini(jd_clean, "job description")
    )

    # Stage 2 – Compute metrics
    cos_sim = compute_cosine_similarity(" ".join(resume_keywords), " ".join(jd_keywords))
    keyword_score = compute_keyword_coverage(resume_keywords, jd_keywords)
    skill_score, matched_skills = compute_skill_match(resume_keywords, jd_keywords)
    ats_score = round((cos_sim + keyword_score + skill_score) / 3, 2)
    repeated_words = compute_repeated_words(resume_text)

    # Stage 3 – Parallel Gemini feedback
    prompts = generate_prompts(resume_text, jd_text)
    feedback_results = await asyncio.gather(*[
        call_gemini(f"""Resume:\n\"\"\"{resume_text}\"\"\"\n\nJD:\n\"\"\"{jd_text}\"\"\"\n\n{prompt}""")
        for prompt in prompts.values()
    ])

    feedback_dict = dict(zip(prompts.keys(), feedback_results))

    # Stage 4 – Output
    output_json = {
        "skills": {
            "resume_skills": list(resume_keywords),
            "jd_skills": list(jd_keywords),
            "matched_skills": matched_skills,
            "missing_skills": list(jd_keywords - resume_keywords)
        },
        "metrics": {
            "cosine_similarity": cos_sim,
            "keyword_coverage": keyword_score,
            "skill_match_score": skill_score,
            "ats_score": ats_score
        },
        "repeated_words": repeated_words,
        "gemini_feedback": feedback_dict
    }

    return output_json

# ─────────────────────────────────────────────
# 🚀 Streamlit UI
st.set_page_config(page_title="Resume Evaluation Results", layout="wide")
st.title("📊 Resume Evaluation Results")

if "resume_text" not in st.session_state or "jd_text" not in st.session_state:
    st.warning("Please upload a resume and job description from Page 1.")
    st.stop()

with st.spinner("Processing in parallel..."):
    results = asyncio.run(main(st.session_state.resume_text, st.session_state.jd_text))
end_time = time.time()
st.success(f"✅ Evaluation completed in {end_time - start_time:.2f} seconds.")
# Display metrics
st.subheader("🔢 Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔁 Cosine Similarity", f"{results['metrics']['cosine_similarity']} %")
col2.metric("🔑 Keyword Coverage", f"{results['metrics']['keyword_coverage']} %")
col3.metric("🛠️ Skill Match Score", f"{results['metrics']['skill_match_score']} %")
col4.metric("📌 ATS Score", f"{results['metrics']['ats_score']} %")

# Visual insights
st.subheader("📈 Visual Insights")
col5, col6 = st.columns(2)
with col5:
    st.markdown("**Keyword Coverage**")
    matched = len(set(results["skills"]["matched_skills"]))
    missing = len(results["skills"]["missing_skills"])
    plot_pie("Keyword Coverage", ["Matched", "Missing"], [matched, missing])

with col6:
    st.markdown("**Skill Match**")
    plot_pie("Skill Match", ["Matched", "Missing"], [matched, missing])

# Repeated words
st.subheader("🔁 Repeated Words (3+ times)")
if results["repeated_words"]:
    for word, count in results["repeated_words"].items():
        st.write(f"• **{word}** appears **{count}** times")
else:
    st.info("✅ No highly repeated words found.")

# Gemini feedback
st.subheader("🤖 Gemini-Powered Feedback")
for section, feedback in results["gemini_feedback"].items():
    st.markdown(f"### 🔹 {section}")
    st.markdown(feedback)
