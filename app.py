"""
app.py — Streamlit Sandbox Demo
--------------------------------
Evaluators upload a small candidates.jsonl (<100 candidates),
and the app runs the full ranking pipeline end-to-end, displaying
results with reasoning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import json
import re
import time
import tempfile
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from datetime import date

# --- Page Config ---
st.set_page_config(
    page_title="AI Candidate Ranker",
    page_icon="🏆",
    layout="wide"
)

# --- Constants ---
ANCHOR_DATE = date.fromisoformat("2024-06-01")

EXCLUDED_JOB_TITLES = [
    "marketing manager", "hr manager", "content writer", "business analyst",
    "project manager", "product manager", "sales", "finance", "accountant"
]
AGENCY_NAMES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra"
]

TARGET_SPEC = (
    "Senior AI Engineer Founding Team Redrob AI Series A talent intelligence platform. "
    "Production experience with embeddings-based retrieval systems, vector databases, and recommendation systems. "
    "Built search and match algorithms deployed to real users. "
    "Strong Python. Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing). "
    "5 to 9 years experience. Shipped real products at meaningful scale."
)

JD_MUST_KEYWORDS = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "hybrid search", "rag", "ndcg", "mrr", "map", "a/b test", "xgboost"
]
PRODUCTION_SIGNALS = ["production", "deployed", "shipped", "real users", "at scale", "serving"]
SNIPPET_RADIUS = 60


# --- Cached Model Loading ---
@st.cache_resource
def load_biencoder():
    """Load bi-encoder model (cached across reruns)."""
    model_path = os.path.join(os.path.dirname(__file__), "model")
    if os.path.exists(model_path):
        return SentenceTransformer(model_path, device='cpu')
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


@st.cache_resource
def load_crossencoder():
    """Load cross-encoder model (cached across reruns)."""
    model_path = os.path.join(os.path.dirname(__file__), "local_cross_encoder")
    if os.path.exists(model_path):
        return CrossEncoder(model_path, device='cpu')
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')


# --- Feature Extraction (from precompute.py) ---
def parse_and_evaluate_profile(applicant):
    prof_data = applicant.get("profile", {})
    work_hist = applicant.get("career_history", [])
    skill_set = applicant.get("skills", [])
    telemetry = applicant.get("redrob_signals", {})

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

    title_lower = str(prof_data.get("current_title", "")).lower()
    orgs = [str(r.get("company", "")).lower() for r in work_hist]

    is_blocked_title = any(rf in title_lower for rf in EXCLUDED_JOB_TITLES)
    agency_count = sum(1 for org in orgs if any(agency in org for agency in AGENCY_NAMES))
    is_pure_agency = (len(orgs) > 0 and agency_count == len(orgs))

    payload_lower = semantic_payload.lower()
    deployment_terms = ["production", "deployed", "shipped", "serving", "real users"]
    has_shipped_code = sum(1 for kw in deployment_terms if kw in payload_lower) > 0

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


# --- Scoring Helpers ---
def calc_experience_weight(y):
    if y < 2: return 0.1
    if y < 5: return 0.5 + 0.5 * (y - 2) / 3
    if y <= 9: return 1.0
    if y <= 12: return 1.0 - 0.3 * (y - 9) / 3
    return 0.4

def calc_availability_weight(d):
    if d <= 30: return 1.0
    if d <= 60: return 0.7
    if d <= 90: return 0.4
    return 0.2


# --- Evidence & Reasoning ---
def collect_evidence_from_raw(candidate):
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    career_text = " ".join((r.get("description", "") or "") for r in career_history).lower()
    skill_names = [s.get("name", "").lower() for s in skills]

    matched_skills, missing_skills = [], []
    for kw in JD_MUST_KEYWORDS:
        if any(kw in s for s in skill_names) or kw in career_text:
            matched_skills.append(kw)
        else:
            missing_skills.append(kw)

    prod_hits = []
    for sig in PRODUCTION_SIGNALS:
        if sig in career_text:
            prod_hits.append(sig)

    return {"matched": list(set(matched_skills)), "missing": list(set(missing_skills)), "production": prod_hits}


def generate_honest_reasoning(candidate, rank, max_len=300):
    if not candidate:
        return "Profile data missing."

    evidence = collect_evidence_from_raw(candidate)
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = (profile.get("current_title") or "Engineer").strip()
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company") or ""
    at_company = f" at {company}" if company else ""

    strengths_part = ""
    if evidence["matched"]:
        strengths_part = f"; strong match on {', '.join(evidence['matched'][:3])}"

    prod_part = ""
    if evidence["production"]:
        prod_part = f"; production evidence ({', '.join(evidence['production'][:2])})"

    open_work = signals.get("open_to_work_flag", False)
    avail_bits = ["actively looking" if open_work else "passive"]
    rr = signals.get("recruiter_response_rate", 0) or 0
    if rr > 0:
        avail_bits.append(f"{rr:.0%} response rate")
    notice = signals.get("notice_period_days", 90) or 90
    if notice < 90:
        avail_bits.append(f"{notice}d notice")

    missing_part = ""
    if len(evidence["missing"]) > 5:
        missing_part = f" Gap: lacks explicit mention of {', '.join(evidence['missing'][:2])}."

    reasoning = f"{yoe}yr {title}{at_company}{strengths_part}{prod_part}; {', '.join(avail_bits)}.{missing_part}"
    if rank > 80:
        reasoning = f"Marginal fit included as filler: {reasoning}"

    reasoning = re.sub(r"[\n\r\t\"]", " ", reasoning)
    reasoning = re.sub(r"\s+", " ", reasoning).strip()
    if len(reasoning) > max_len:
        reasoning = reasoning[:max_len - 3] + "..."
    return reasoning


# --- Main Pipeline ---
def run_pipeline(raw_candidates, progress_bar):
    """Run the full ranking pipeline on uploaded candidates."""

    # Stage 0: Parse profiles
    progress_bar.progress(5, "Parsing candidate profiles...")
    parsed_data = [parse_and_evaluate_profile(c) for c in raw_candidates]
    df_meta = pd.DataFrame(parsed_data)
    n_docs = len(df_meta)

    # Stage 1: Dense Retrieval
    progress_bar.progress(15, "Stage 1: Dense semantic retrieval...")
    encoder = load_biencoder()

    dense_matrix = encoder.encode(
        df_meta['semantic_payload'].tolist(),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    dim_size = dense_matrix.shape[1]
    faiss_db = faiss.IndexFlatL2(dim_size)
    faiss.normalize_L2(dense_matrix)
    faiss_db.add(dense_matrix)

    spec_vec = encoder.encode([TARGET_SPEC], convert_to_numpy=True)
    faiss.normalize_L2(spec_vec)

    dists, idxs = faiss_db.search(spec_vec, n_docs)
    dense_scores = np.zeros(n_docs)
    for p, orig_idx in enumerate(idxs[0]):
        dense_scores[orig_idx] = 1 - (dists[0][p] / 2)
    df_meta['dense_sim'] = dense_scores

    df_meta['exp_w'] = df_meta['years_of_experience'].apply(calc_experience_weight)
    df_meta['avail_w'] = df_meta['notice_period_days'].apply(calc_availability_weight)

    # Stage 2: Hard Filtering
    progress_bar.progress(35, "Stage 2: Disqualifier & trap filtering...")
    df_meta['base_penalty'] = 1.0
    df_meta.loc[df_meta['is_honeypot'] == True, 'base_penalty'] *= 0.01
    df_meta.loc[df_meta['is_pure_consulting'] == True, 'base_penalty'] *= 0.1
    df_meta.loc[df_meta['is_red_flag_title'] == True, 'base_penalty'] *= 0.1
    df_meta.loc[df_meta['is_job_hopper'] == True, 'base_penalty'] *= 0.2
    df_meta.loc[df_meta['is_pure_research'] == True, 'base_penalty'] *= 0.01
    df_meta.loc[df_meta['langchain_trap'] == True, 'base_penalty'] *= 0.2
    df_meta.loc[(df_meta['days_offline'] > 180) & (df_meta['recruiter_response_rate'] < 0.15), 'base_penalty'] *= 0.1

    df_meta['s1_score'] = df_meta['base_penalty'] * ((df_meta['exp_w'] * 0.20) + (0.80 * df_meta['dense_sim']))

    # Stage 3: Cross-Encoder Reranking
    progress_bar.progress(50, "Stage 3: Cross-encoder reranking...")
    top_n = min(250, len(df_meta))
    candidate_subset = df_meta.sort_values(by='s1_score', ascending=False).head(top_n).copy()

    cross_model = load_crossencoder()
    seq_pairs = [[TARGET_SPEC, t] for t in candidate_subset['semantic_payload'].fillna("").tolist()]
    seq_scores = cross_model.predict(seq_pairs)
    candidate_subset['cross_sim'] = (seq_scores - np.min(seq_scores)) / (np.max(seq_scores) - np.min(seq_scores) + 1e-9)

    # Stage 4: RRF
    progress_bar.progress(70, "Stage 4: Reciprocal Rank Fusion...")
    candidate_subset['r_dense'] = candidate_subset['dense_sim'].rank(ascending=False)
    candidate_subset['r_cross'] = candidate_subset['cross_sim'].rank(ascending=False)
    candidate_subset['fusion_score'] = (1.0 / (60 + candidate_subset['r_dense'])) + (1.0 / (60 + candidate_subset['r_cross']))
    max_rrf = 0.03278688
    candidate_subset['fusion_score_norm'] = (candidate_subset['fusion_score'] / max_rrf).clip(upper=1.0)

    # Stage 5: Final Blending
    progress_bar.progress(85, "Stage 5: Final heuristic blending...")
    candidate_subset['surgical_boost'] = 0.0
    candidate_subset.loc[candidate_subset['github_activity_score'] > 50, 'surgical_boost'] += 0.03
    candidate_subset.loc[candidate_subset['saved_by_recruiters_30d'] > 5, 'surgical_boost'] += 0.02

    candidate_subset['raw_metric'] = (
        (candidate_subset['exp_w'] * 0.15) +
        (candidate_subset['avail_w'] * 0.10) +
        (0.75 * candidate_subset['fusion_score_norm'])
    ) + candidate_subset['surgical_boost']

    candidate_subset['final_computation'] = candidate_subset['raw_metric'] * candidate_subset['base_penalty']
    candidate_subset['rounded_final'] = candidate_subset['final_computation'].round(4)

    result_count = min(100, len(candidate_subset))
    top_tier = candidate_subset.sort_values(by=['rounded_final', 'candidate_id'], ascending=[False, True]).head(result_count).copy()
    top_tier['rank'] = range(1, result_count + 1)

    # Generate reasoning
    progress_bar.progress(95, "Generating evidence-based reasoning...")
    raw_map = {c["candidate_id"]: c for c in raw_candidates if c.get("candidate_id") in set(top_tier['candidate_id'].tolist())}
    top_tier['reasoning'] = top_tier.apply(
        lambda row: generate_honest_reasoning(raw_map.get(row['candidate_id'], {}), row['rank']),
        axis=1
    )

    final_output = top_tier[['candidate_id', 'rank', 'rounded_final', 'reasoning']].rename(columns={'rounded_final': 'score'})
    progress_bar.progress(100, "Done!")
    return final_output


# ===================== STREAMLIT UI =====================

# Header
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="font-size: 2.5rem; margin-bottom: 0.2rem;">🏆 Thala For A Reason</h1>
    <p style="font-size: 1.2rem; color: #888;">Intelligent Candidate Discovery & Ranking — India Runs Data & AI Challenge</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Pipeline Overview
with st.expander("📋 Pipeline Architecture", expanded=False):
    st.markdown("""
    | Stage | Method | Description |
    |-------|--------|-------------|
    | 1 | Dense Retrieval | Bi-encoder (MiniLM) + FAISS cosine similarity |
    | 2 | Hard Filtering | Disqualify honeypots, blocked titles, agency-only |
    | 3 | Cross-Encoder | Rerank with ms-marco cross-encoder |
    | 4 | RRF | Reciprocal Rank Fusion of dual rankings |
    | 5 | Heuristic Blend | Experience, availability, GitHub, recruiter saves |
    """)

# Job Description
with st.expander("🎯 Target Job Description", expanded=False):
    st.info(TARGET_SPEC)

st.divider()

# File Upload
st.subheader("📁 Upload Candidates")
uploaded_file = st.file_uploader(
    "Upload a `candidates.jsonl` file (one JSON object per line, max ~100 candidates)",
    type=["jsonl", "json"],
    help="Each line should be a valid JSON object matching the candidate schema."
)

if uploaded_file is not None:
    # Parse JSONL
    raw_text = uploaded_file.read().decode("utf-8")
    raw_candidates = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                raw_candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Handle JSON array format too
    if len(raw_candidates) == 0:
        try:
            raw_candidates = json.loads(raw_text)
        except json.JSONDecodeError:
            st.error("Could not parse the uploaded file. Ensure it's valid JSONL or JSON.")

    if raw_candidates:
        st.success(f"✅ Loaded **{len(raw_candidates)}** candidates")

        if st.button("🚀 Run Ranking Pipeline", type="primary", use_container_width=True):
            t0 = time.time()
            progress_bar = st.progress(0, "Initializing...")

            results = run_pipeline(raw_candidates, progress_bar)

            elapsed = time.time() - t0
            st.balloons()

            # Results
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Candidates Ranked", len(results))
            col2.metric("Top Score", f"{results['score'].max():.4f}")
            col3.metric("Pipeline Time", f"{elapsed:.1f}s")

            st.divider()
            st.subheader("📊 Ranked Results")

            # Display table
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                    "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
                }
            )

            # Download button
            csv_data = results.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Submission CSV",
                data=csv_data,
                file_name="Thala_For_A_Reason.csv",
                mime="text/csv",
                use_container_width=True
            )
else:
    st.info("👆 Upload a `candidates.jsonl` file to get started.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem;">
    Built by <strong>Thala For A Reason</strong> | Piyushkanta Panda | India Runs Data & AI Challenge 2026
</div>
""", unsafe_allow_html=True)
