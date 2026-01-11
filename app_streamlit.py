import streamlit as st
import requests
import pandas as pd

API_ANALYZE_URL = "http://localhost:8000/analyze"
API_RANK_URL = "http://localhost:8000/rank_resumes"

st.set_page_config(page_title="AI Job Intelligence System", layout="wide")
st.title(" AI Job Application Intelligence System (Enterprise Mode)")

tab1, tab2 = st.tabs([" Job Seeker Mode", " Recruiter Mode"])


 
with tab1:
    st.header(" Job Seeker Mode")
    st.write("Upload your resume + paste job description to get fit score and improvements.")

    jd = st.text_area(" Job Description", height=200, key="jd_seeker")
    resume = st.file_uploader(" Upload Resume PDF", type=["pdf"], key="resume_seeker")

    if st.button("Analyze Resume", key="analyze_btn"):
        if not jd or not resume:
            st.error("Please provide both Job Description and Resume PDF.")
        else:
            with st.spinner("Analyzing..."):
                data = {"job_description": jd}
                response = requests.post(API_ANALYZE_URL, data=data, files={"resume_file": resume})

                if response.status_code != 200:
                    st.error("API Error: " + response.text)
                else:
                    result = response.json()
                    st.success(f" Hybrid Match Score: {result['match_score']} / 100")
                    st.info(f" Fit Prediction Score: {result['fit_prediction_score']} / 100")
                    st.json(result)


with tab2:
    st.header(" Recruiter Mode")
    st.write("Upload multiple resumes (up to 50 PDFs). System ranks the best candidates based on the job description.")

    jd_recruiter = st.text_area(" Job Description", height=200, key="jd_recruiter")
    resumes = st.file_uploader(" Upload Resume PDFs", type=["pdf"], accept_multiple_files=True, key="resumes_bulk")

    top_k = st.slider("Select Top K resumes to display", min_value=5, max_value=50, value=10)

    if st.button("Rank Resumes", key="rank_btn"):
        if not jd_recruiter or not resumes:
            st.error("Please provide Job Description and upload at least 1 resume.")
        else:
            if len(resumes) > 50:
                st.error("Max 50 resumes allowed.")
            else:
                with st.spinner("Ranking resumes... (this may take time)"):
                    data = {"job_description": jd_recruiter, "top_k": str(top_k)}

                    files = [("resume_files", (r.name, r, "application/pdf")) for r in resumes]

                    response = requests.post(API_RANK_URL, data=data, files=files)

                    if response.status_code != 200:
                        st.error("API Error: " + response.text)
                    else:
                        result = response.json()
                        ranked = result["ranked_resumes"]

                        st.success(f" Ranked {result['total_resumes']} resumes. Showing top {result['top_k']}.")

                        
                        table_data = []
                        for i, row in enumerate(ranked, start=1):
                            table_data.append({
                                "Rank": i,
                                "Resume Name": row["resume_name"],
                                "Fit Score": row["fit_prediction_score"],
                                "Match Score": row["match_score"],
                                "Missing Skills": ", ".join(row["missing_skills"][:5]),
                                "Missing Keywords": ", ".join(row["missing_keywords"][:5])
                            })

                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)

                       
                        st.subheader(" Full Candidate Details")
                        for row in ranked:
                            with st.expander(f"{row['resume_name']} (Fit: {row['fit_prediction_score']} | Match: {row['match_score']})"):
                                st.write("### Missing Skills")
                                st.write(row["missing_skills"])

                                st.write("### Missing Keywords")
                                st.write(row["missing_keywords"])

                                st.write("### Similarity Breakdown")
                                st.json(row["similarity"])
