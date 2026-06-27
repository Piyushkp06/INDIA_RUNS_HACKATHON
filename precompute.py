"""
precompute.py
-------------
Offline pre-computation step: reads candidates.jsonl, extracts structured
features, builds dense embeddings via SentenceTransformer, and writes a FAISS
index + Parquet metadata file to disk.

These artifacts are consumed by `rank.py` at inference time.

Usage:
    python precompute.py --candidates ./Data/India_runs_data_and_ai_challenge/candidates.jsonl
"""

import json
import argparse
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import date
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
ANCHOR_DATE = date.fromisoformat("2024-06-01")

EXCLUDED_JOB_TITLES = [
    "marketing manager", "hr manager", "content writer", "business analyst",
    "project manager", "product manager", "sales", "finance", "accountant"
]
AGENCY_NAMES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra"
]

# Output artifact paths (written to repo root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_OUT = os.path.join(SCRIPT_DIR, "applicant_space.faiss")
PARQUET_OUT = os.path.join(SCRIPT_DIR, "applicant_metadata.parquet")


def yield_applicant_records(file_path):
    """Stream candidate records from a JSONL file."""
    with open(file_path, "rt", encoding="utf-8") as file_obj:
        for row in file_obj:
            if row.strip():
                yield json.loads(row)


def parse_and_evaluate_profile(applicant):
    """Extract structured features and heuristic flags from a single candidate."""
    prof_data = applicant.get("profile", {})
    work_hist = applicant.get("career_history", [])
    skill_set = applicant.get("skills", [])
    telemetry = applicant.get("redrob_signals", {})

    # --- 1. Semantic Narrative Construction ---
    high_value_context = []

    title = prof_data.get('current_title', '')
    headline = prof_data.get('headline', '')
    high_value_context.append(f"{title}. {headline}")

    expert_skills = [s.get('name', '') for s in skill_set if s.get('proficiency') in ["expert", "advanced"]]
    if expert_skills:
        high_value_context.append("Expertise: " + ", ".join(expert_skills[:15]))

    for role in work_hist[:3]:
        role_title = role.get('title', '')
        role_desc = role.get("description", "").replace("\n", ". ")[:500]
        high_value_context.append(f"Role: {role_title}. {role_desc}")

    combined_text = " | ".join(high_value_context).strip()
    words = combined_text.split()
    semantic_payload = " ".join(words[:350])

    # --- 2. Base Heuristics & Exclusions ---
    title_lower = str(prof_data.get("current_title", "")).lower()
    orgs = [str(r.get("company", "")).lower() for r in work_hist]

    is_blocked_title = any(rf in title_lower for rf in EXCLUDED_JOB_TITLES)
    agency_count = sum(1 for org in orgs if any(agency in org for agency in AGENCY_NAMES))
    is_pure_agency = (len(orgs) > 0 and agency_count == len(orgs))

    payload_lower = semantic_payload.lower()
    deployment_terms = ["production", "deployed", "shipped", "serving", "real users"]
    has_shipped_code = sum(1 for kw in deployment_terms if kw in payload_lower) > 0

    # --- 3. JD Disqualifiers & Behavioral Traps ---
    claimed_yoe = prof_data.get("years_of_experience", 0)
    calculated_months = sum(r.get("duration_months", 0) for r in work_hist)

    num_companies = len(set([org for org in orgs if org]))
    is_job_hopper = False
    if num_companies > 0:
        avg_tenure_years = (calculated_months / 12) / num_companies
        if claimed_yoe > 3 and avg_tenure_years < 1.5:
            is_job_hopper = True

    research_terms = ["postdoc", "research assistant", "phd candidate", "research scientist"]
    is_pure_research = any(rt in title_lower for rt in research_terms) and not has_shipped_code

    ai_skills = [s for s in skill_set if "langchain" in s.get("name", "").lower() or "openai" in s.get("name", "").lower()]
    langchain_trap = len(ai_skills) > 0 and all(s.get("duration_months", 0) < 12 for s in ai_skills) and claimed_yoe < 3

    # --- 4. Behavioral Signals Processing ---
    try:
        last_active = date.fromisoformat(telemetry.get("last_active_date", "2000-01-01"))
        days_offline = (ANCHOR_DATE - last_active).days
    except Exception:
        days_offline = 999

    resp_rate = telemetry.get("recruiter_response_rate", 0.0)
    apps_30d = telemetry.get("applications_submitted_30d", 0)
    interview_completion = telemetry.get("interview_completion_rate", 1.0)
    gh_score = telemetry.get("github_activity_score", -1)
    linkedin_conn = telemetry.get("linkedin_connected", False)

    is_synthetic_trap = False
    unverified_expert_skills = [s["name"] for s in skill_set if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0]
    max_skill_months = max([s.get("duration_months", 0) for s in skill_set]) if skill_set else 0

    if len(unverified_expert_skills) >= 3:
        is_synthetic_trap = True
    elif calculated_months > 0 and abs((calculated_months / 12) - claimed_yoe) > 5:
        is_synthetic_trap = True
    elif max_skill_months > (claimed_yoe * 12) + 24:
        is_synthetic_trap = True
    elif apps_30d > 10 and interview_completion == 0.0:
        is_synthetic_trap = True
    elif resp_rate > 0.9 and days_offline > 365:
        is_synthetic_trap = True
    elif claimed_yoe > 8 and gh_score == -1 and not linkedin_conn:
        is_synthetic_trap = True

    return {
        "candidate_id": applicant.get("candidate_id"),
        "semantic_payload": semantic_payload,
        "title_snapshot": title,
        "years_of_experience": claimed_yoe,
        "notice_period_days": telemetry.get("notice_period_days", 90),
        "recruiter_response_rate": resp_rate,
        "days_offline": days_offline,
        "github_activity_score": gh_score,
        "profile_views_30d": telemetry.get("profile_views_received_30d", 0),
        "saved_by_recruiters_30d": telemetry.get("saved_by_recruiters_30d", 0),
        "is_red_flag_title": is_blocked_title,
        "is_pure_consulting": is_pure_agency,
        "has_production_ml": has_shipped_code,
        "is_job_hopper": is_job_hopper,
        "is_pure_research": is_pure_research,
        "langchain_trap": langchain_trap,
        "is_honeypot": is_synthetic_trap
    }


def construct_applicant_dataset(file_path):
    """Parse all candidate records and return a structured DataFrame."""
    print("Extracting records & generating structural features...")
    parsed_data = [parse_and_evaluate_profile(a) for a in yield_applicant_records(file_path)]
    return pd.DataFrame(parsed_data)


def create_vector_store(dataframe):
    """Encode semantic payloads and build a normalized FAISS L2 index."""
    print("Initializing Dense Embedding Matrix...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    dense_matrix = encoder.encode(
        dataframe['semantic_payload'].tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dim_size = dense_matrix.shape[1]
    faiss_db = faiss.IndexFlatL2(dim_size)
    faiss.normalize_L2(dense_matrix)
    faiss_db.add(dense_matrix)
    return faiss_db


def main():
    parser = argparse.ArgumentParser(description="Pre-compute FAISS index and metadata for candidate ranking.")
    parser.add_argument('--candidates', type=str, required=True, help="Path to candidates.jsonl")
    args = parser.parse_args()

    df = construct_applicant_dataset(args.candidates)
    index_db = create_vector_store(df)

    print("Writing artifacts to disk...")
    faiss.write_index(index_db, FAISS_OUT)
    df.to_parquet(PARQUET_OUT, engine='pyarrow')
    print(f"  -> {FAISS_OUT}")
    print(f"  -> {PARQUET_OUT}")
    print("Offline pre-computation complete. You are ready for live inference.")


if __name__ == "__main__":
    main()
