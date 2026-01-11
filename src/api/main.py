from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from ..utils.pdf import extract_text_from_pdf
from ..scoring.score_engine import ScoreEngine
from ..scoring.schema import AnalysisResponse

from ..scoring.rank_engine import RankEngine
from ..scoring.rank_schema import RankResponse


app = FastAPI(title="AI Job Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ScoreEngine()
rank_engine = RankEngine()


 
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...)
):
    pdf_bytes = await resume_file.read()
    resume_text = extract_text_from_pdf(pdf_bytes)

    result = engine.analyze(job_description, resume_text)
    return result


 
@app.post("/rank_resumes", response_model=RankResponse)
async def rank_resumes(
    job_description: str = Form(...),
    resume_files: List[UploadFile] = File(...),
    top_k: int = Form(10)
):
    resumes = []

    for file in resume_files:
        pdf_bytes = await file.read()
        resume_text = extract_text_from_pdf(pdf_bytes)

        resumes.append({
            "name": file.filename,
            "text": resume_text
        })

    ranked_top, _ = rank_engine.rank(job_description, resumes, top_k=top_k)

    return {
        "job_description": job_description[:400] + "..." if len(job_description) > 400 else job_description,
        "total_resumes": len(resumes),
        "top_k": top_k,
        "ranked_resumes": ranked_top
    }
