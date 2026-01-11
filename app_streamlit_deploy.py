import streamlit as st
import pandas as pd

from src.utils.pdf import extract_text_from_pdf
from src.scoring.score_engine import ScoreEngine

st.set_page_config(page_title="AI Job Intelligence", layout="wide")
st.title(" AI Job Application Intelligence System (Deployed)")
st.caption("This version runs everything inside Streamlit. No FastAPI needed.")

@st.cache_resource
def get_engine():
    return ScoreEngine()

engine = get_engine()

tab1, tab2 = st.tabs([" Job Seeker Mode", " Recruiter Mode"])

with tab1:
    st.subheader(" Job Seeker Mode")
    jd = st.text_area(" Job Description", height=220)
    resume = st.file_uploader(" Upload Resume PDF", type=["pdf"])

    if st.button("Analyze Resume"):
        if not jd or not resume:
            st.error("Please provide both Job Description and Resume PDF.")
        else:
            with st.spinner("Analyzing..."):
                pdf_bytes = resume.read()
                resume_text = extract_text_from_pdf(pdf_bytes)
                result = engine.analyze(jd, resume_text)

                st.success(f" Match Score: {result['match_score']} / 100")
                st.info(f" Fit Score: {result['fit_prediction_score']} / 100")
                st.json(result)

with tab2:
    st.subheader(" Recruiter Mode")
    jd2 = st.text_area(" Job Description (Recruiter)", height=220)
    resumes = st.file_uploader(" Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
    top_k = st.slider("Top K", 5, 50, 10)

    if st.button("Rank Resumes"):
        if not jd2 or not resumes:
            st.error("Please provide Job Description and at least 1 resume.")
        elif len(resumes) > 50:
            st.error("Max 50 resumes allowed.")
        else:
            with st.spinner("Ranking..."):
                rows = []
                for f in resumes:
                    resume_text = extract_text_from_pdf(f.read())
                    analysis = engine.analyze(jd2, resume_text)

                    rows.append({
                        "Resume": f.name,
                        "Fit Score": analysis["fit_prediction_score"],
                        "Match Score": analysis["match_score"],
                        "Missing Skills": ", ".join(analysis["skills"]["missing"][:5]),
                    })

                rows.sort(key=lambda x: (x["Fit Score"], x["Match Score"]), reverse=True)
                df = pd.DataFrame(rows[:top_k])
                st.dataframe(df, use_container_width=True)
