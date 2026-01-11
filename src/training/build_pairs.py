import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/resume_data_for_ranking.csv")
OUT_PATH = Path("data/processed/train_pairs.csv")

def safe_join(row, cols):
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            parts.append(str(row[c]))
    return " ".join(parts).strip()

def main():
    print(f"✅ Loading dataset: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    print("✅ Rows:", len(df))
    print("✅ Columns:", df.columns.tolist())

    # ✅ Resume fields -> merged into resume_text
    resume_cols = [
        "career_objective",
        "skills",
        "educational_institution_name",
        "degree_names",
        "major_field_of_studies",
        "professional_company_names",
        "positions",
        "responsibilities",
        "extra_curricular_activity_types",
        "role_positions",
        "languages",
        "certification_skills"
    ]

    # ✅ Job fields -> merged into job_description
    job_cols = [
        "job_position_name",
        "educationaL_requirements",
        "experiencere_requirement",
        "responsibilities.1",
        "skills_required"
    ]

    # ✅ Label column
    label_col = "matched_score"

    # Drop rows with missing label
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # Build combined text fields
    df["resume_text"] = df.apply(lambda r: safe_join(r, resume_cols), axis=1)
    df["job_description"] = df.apply(lambda r: safe_join(r, job_cols), axis=1)

    # Drop empty rows
    df = df[(df["resume_text"].str.len() > 20) & (df["job_description"].str.len() > 20)]

    # ✅ Convert matched_score to binary label
    # You can tune this threshold later.
    # For now: >= 0.6 means fit
    df["label"] = df[label_col].apply(lambda x: 1 if x >= 0.6 else 0)

    out_df = df[["resume_text", "job_description", "label", label_col]].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"✅ Saved: {OUT_PATH}")
    print("✅ Label Distribution:")
    print(out_df["label"].value_counts(normalize=True))

    print("\n✅ Sample rows:")
    print(out_df.head(2))

if __name__ == "__main__":
    main()
