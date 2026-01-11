# AI Job Intelligence System

# AI Job Application Intelligence System (NLP + ML + Explainability)

Live Demo (Hugging Face): https://huggingface.co/spaces/rubel10/ai-job-intelligence  
GitHub Repo: https://github.com/rubel-dev/ai_job_intelligence

## What this project solves
Recruiters and job seekers waste time because:
- Recruiters receive too many resumes per job and keyword-only filtering misses strong candidates.
- Job seekers apply blindly and don’t know what to improve.

This system bridges that gap by scoring resume–job fit, explaining why, and suggesting improvements.

---

## Key Features

### 👤 Job Seeker Mode
- Upload a resume (PDF) + paste a job description
- Returns:
  - **Hybrid Match Score (0–100)**
  - **Fit Prediction Score (0–100)** (ML probability)
  - **Skill overlap + missing skills**
  - **Keyword optimization suggestions**
  - **Explainability output**
  - **Actionable recommendations**

### 🏢 Recruiter Mode
- Paste a job description
- Upload multiple resumes (up to 50 PDFs)
- Produces a ranked shortlist table:
  - Fit Score + Match Score
  - Missing skills
  - Quick comparison for shortlisting

---

## How it works (Architecture)

**Resume PDF → Text Extraction → NLP Cleaning → Skill Extraction → Similarity Models → Feature Engineering → Fit Classifier → Explainability + Recommendations**

### Scoring & Modeling (Hybrid Approach)
- **TF-IDF similarity** → strong for keyword relevance (ATS-like behavior)
- **Sentence-BERT similarity** → captures semantic meaning (synonyms/context)
- **Skill overlap** → ensures practical relevance

### Fit Prediction (ML)
- A classifier predicts fit probability based on engineered features:
  - TF-IDF similarity  
  - SBERT similarity  
  - skill overlap percent  
  - missing skill count  
  - keyword match count  

### Explainability
- Human-readable feature contributions + optional SHAP explanations (if enabled in your pipeline)

---

## Tech Stack
- **Python**
- **Streamlit** (UI)
- **Scikit-learn** (classifier)
- **Sentence-Transformers (SBERT)** (semantic embeddings)
- **SHAP** (explainability)
- Pandas, NumPy
- Hugging Face Spaces (deployment)

---

## Run Locally

### 1) Clone the repo
```bash
git clone https://github.com/rubel-dev/ai_job_intelligence.git
cd ai_job_intelligence
2) Create & activate environment (recommended)
Using conda:

bash
Copy code
conda create -n jobint python=3.10 -y
conda activate jobint
3) Install dependencies
bash
Copy code
pip install -r requirements.txt
4) Run the app
If your main Streamlit file is app.py:

bash
Copy code
streamlit run app.py
Or if your file is app_streamlit_deploy.py:

bash
Copy code
streamlit run app_streamlit_deploy.py
Dataset (Training)
This project can use real datasets from Kaggle for resume/job data.
You can:

Train classifier on labeled/weak-labeled resume–JD pairs

Save model with joblib

Reuse model in the deployed pipeline

Roadmap (Next Improvements)
Fine-tune SBERT on real resume–JD pairs (domain adaptation)

Use NER (spaCy/transformer) for stronger skill extraction

Embedding caching for faster recruiter ranking

Add section-aware parsing (Experience / Skills / Projects)

Add bias checks and calibration for fair scoring

Demo
 Hugging Face App: https://huggingface.co/spaces/rubel10/ai-job-intelligence

Author
Rubel Mia
GitHub: https://github.com/rubel-dev
Project Repo: https://github.com/rubel-dev/ai_job_intelligence
 
