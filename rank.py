"""
rank.py
-------
Inference pipeline: loads pre-computed FAISS index + metadata, runs a
multi-stage ranking pipeline against the target job description, and
outputs a submission CSV with the top 100 candidates.

Usage:
    python rank.py --candidates ./Data/India_runs_data_and_ai_challenge/candidates.jsonl --out ./team_xxx.csv
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
import warnings
import argparse
import os
import sys
import json
import re

warnings.filterwarnings("ignore")

# --- Configuration (Relative to script location for portability) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_STORE = os.path.join(SCRIPT_DIR, "applicant_metadata.parquet")
FAISS_STORE = os.path.join(SCRIPT_DIR, "applicant_space.faiss")

# Pointing to local offline models to avoid network calls during ranking
BI_MODEL = os.path.join(SCRIPT_DIR, "model")
CROSS_MODEL = os.path.join(SCRIPT_DIR, "local_cross_encoder")

# The core target job description string
TARGET_SPEC = (
    "Senior AI Engineer Founding Team Redrob AI Series A talent intelligence platform. "
    "Production experience with embeddings-based retrieval systems, vector databases, and recommendation systems. "
    "Built search and match algorithms deployed to real users. "
    "Strong Python. Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing). "
    "5 to 9 years experience. Shipped real products at meaningful scale."
)

# --- Reasoning Configurations ---
JD_MUST_KEYWORDS = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "hybrid search", "rag", "ndcg", "mrr", "map", "a/b test", "xgboost"
]
PRODUCTION_SIGNALS = ["production", "deployed", "shipped", "real users", "at scale", "serving"]
SNIPPET_RADIUS = 60


def calc_experience_weight(y):
    """Bell-curve weight: peaks at 5-9 years, tapers outside."""
    if y < 2: return 0.1
    if y < 5: return 0.5 + 0.5 * (y - 2) / 3
    if y <= 9: return 1.0
    if y <= 12: return 1.0 - 0.3 * (y - 9) / 3
    return 0.4


def calc_availability_weight(d):
    """Availability decay: shorter notice period = higher weight."""
    if d <= 30: return 1.0
    if d <= 60: return 0.7
    if d <= 90: return 0.4
    return 0.2


# --- Evidence-Based Reasoning Generators ---

def _snippet_around(text: str, keyword: str) -> str:
    """Extracts a readable snippet of text around a keyword match."""
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return ""
    idx = m.start()
    start = max(0, idx - SNIPPET_RADIUS)
    end = min(len(text), m.end() + SNIPPET_RADIUS)
    snippet = text[start:end].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return f"…{snippet}…" if start > 0 else f"{snippet}…"


def collect_evidence_from_raw(candidate: dict) -> dict:
    """Matches raw candidate JSON against JD keywords for citable reasoning."""
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Build text blobs
    career_text = " ".join((r.get("description", "") or "") for r in career_history).lower()
    skill_names = [s.get("name", "").lower() for s in skills]

    matched_skills = []
    missing_skills = []

    # 1. Match core keywords against skills and text
    for kw in JD_MUST_KEYWORDS:
        if any(kw in s for s in skill_names):
            matched_skills.append(kw)
        elif kw in career_text:
            matched_skills.append(kw)
        else:
            missing_skills.append(kw)

    # 2. Extract Production Evidence
    prod_hits = []
    for sig in PRODUCTION_SIGNALS:
        if sig in career_text:
            prod_hits.append({
                "keyword": sig,
                "snippet": _snippet_around(career_text, sig)
            })

    return {
        "matched": list(set(matched_skills)),
        "missing": list(set(missing_skills)),
        "production": prod_hits
    }


def generate_honest_reasoning(candidate: dict, rank: int, max_len: int = 300) -> str:
    """Honest one-liner citing only what the profile actually shows."""
    if not candidate:
        return "Profile data missing."

    evidence = collect_evidence_from_raw(candidate)
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = (profile.get("current_title") or "Engineer").strip()
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company") or ""
    at_company = f" at {company}" if company else ""

    # Synthesize Strengths
    strengths_part = ""
    if evidence["matched"]:
        strengths = ", ".join(evidence["matched"][:3])
        strengths_part = f"; strong match on {strengths}"

    # Synthesize Production Proof
    prod_part = ""
    if evidence["production"]:
        kws = ", ".join(dict.fromkeys(h["keyword"] for h in evidence["production"][:2]))
        prod_part = f"; production evidence ({kws})"

    # Synthesize Logistics / Availability
    open_work = signals.get("open_to_work_flag", False)
    avail_bits = ["actively looking" if open_work else "passive"]

    rr = signals.get("recruiter_response_rate", 0) or 0
    if rr > 0:
        avail_bits.append(f"{rr:.0%} response rate")

    notice = signals.get("notice_period_days", 90) or 90
    if notice < 90:
        avail_bits.append(f"{notice}d notice")

    # Synthesize Gaps (Honest concerns)
    missing_part = ""
    if len(evidence["missing"]) > 5:
        gaps = ", ".join(evidence["missing"][:2])
        missing_part = f" Gap: lacks explicit mention of {gaps}."

    # Final string assembly
    reasoning = f"{yoe}yr {title}{at_company}{strengths_part}{prod_part}; {', '.join(avail_bits)}.{missing_part}"

    # Add filler explanation if ranked very low (maintains rank consistency)
    if rank > 80:
        reasoning = f"Marginal fit included as filler: {reasoning}"

    # Cleanup
    reasoning = re.sub(r"[\n\r\t\"]", " ", reasoning)
    reasoning = re.sub(r"\s+", " ", reasoning).strip()

    if len(reasoning) > max_len:
        reasoning = reasoning[: max_len - 3] + "..."

    return reasoning


def execute_scoring_pipeline(candidates_file, output_csv):
    """Run the full 5-stage ranking pipeline and write submission CSV."""
    t0 = time.time()
    print("Booting Inference Pipeline...")

    # Pre-computation check
    if not os.path.exists(PARQUET_STORE):
        print(f"Error: Pre-computed file not found: {PARQUET_STORE}", file=sys.stderr)
        print("Run `python precompute.py --candidates <path>` first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(FAISS_STORE):
        print(f"Error: Pre-computed file not found: {FAISS_STORE}", file=sys.stderr)
        print("Run `python precompute.py --candidates <path>` first.", file=sys.stderr)
        sys.exit(1)

    df_meta = pd.read_parquet(PARQUET_STORE)
    db_faiss = faiss.read_index(FAISS_STORE)

    print(">> Stage 1: Dense Semantic Pass...")
    model_dense = SentenceTransformer(BI_MODEL, device='cpu')
    spec_vec = model_dense.encode([TARGET_SPEC], convert_to_numpy=True)
    faiss.normalize_L2(spec_vec)

    n_docs = len(df_meta)
    dists, idxs = db_faiss.search(spec_vec, n_docs)

    dense_scores = np.zeros(n_docs)
    for p, orig_idx in enumerate(idxs[0]):
        dense_scores[orig_idx] = 1 - (dists[0][p] / 2)

    df_meta['dense_sim'] = dense_scores

    # Base feature weights
    df_meta['exp_w'] = df_meta['years_of_experience'].apply(calc_experience_weight)
    df_meta['avail_w'] = df_meta['notice_period_days'].apply(calc_availability_weight)

    print(">> Stage 2: Hard Disqualifier & Trap Filtering...")
    df_meta['base_penalty'] = 1.0
    df_meta.loc[df_meta['is_honeypot'] == True, 'base_penalty'] *= 0.01
    df_meta.loc[df_meta['is_pure_consulting'] == True, 'base_penalty'] *= 0.1
    df_meta.loc[df_meta['is_red_flag_title'] == True, 'base_penalty'] *= 0.1
    df_meta.loc[df_meta['is_job_hopper'] == True, 'base_penalty'] *= 0.2
    df_meta.loc[df_meta['is_pure_research'] == True, 'base_penalty'] *= 0.01
    df_meta.loc[df_meta['langchain_trap'] == True, 'base_penalty'] *= 0.2

    # Behavioral Tank
    df_meta.loc[(df_meta['days_offline'] > 180) & (df_meta['recruiter_response_rate'] < 0.15), 'base_penalty'] *= 0.1

    df_meta['s1_score'] = df_meta['base_penalty'] * ((df_meta['exp_w'] * 0.20) + (0.80 * df_meta['dense_sim']))

    print(">> Stage 3: Deep Contextual Sequence Attention...")
    candidate_subset = df_meta.sort_values(by='s1_score', ascending=False).head(250).copy()

    model_cross = CrossEncoder(CROSS_MODEL, device='cpu')
    seq_pairs = [[TARGET_SPEC, t] for t in candidate_subset['semantic_payload'].fillna("").tolist()]

    seq_scores = model_cross.predict(seq_pairs)
    candidate_subset['cross_sim'] = (seq_scores - np.min(seq_scores)) / (np.max(seq_scores) - np.min(seq_scores) + 1e-9)

    print(">> Stage 4: 2-Way Reciprocal Rank Fusion...")
    candidate_subset['r_dense'] = candidate_subset['dense_sim'].rank(ascending=False)
    candidate_subset['r_cross'] = candidate_subset['cross_sim'].rank(ascending=False)

    candidate_subset['fusion_score'] = (1.0 / (60 + candidate_subset['r_dense'])) + (1.0 / (60 + candidate_subset['r_cross']))
    max_theoretical_rrf = 0.03278688
    candidate_subset['fusion_score_norm'] = (candidate_subset['fusion_score'] / max_theoretical_rrf).clip(upper=1.0)

    print(">> Stage 5: Final Heuristic Blending...")
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

    print(">> Compiling Strict Submissions...")
    # Tie-breaking logic: candidate_id ascending for equal scores
    top_tier = candidate_subset.sort_values(by=['rounded_final', 'candidate_id'], ascending=[False, True]).head(100).copy()
    top_tier['rank'] = range(1, 101)

    # --- Raw JSON Extraction for Evidence-Based Reasoning ---
    print(">> Extracting Raw Evidence for Top 100 Candidates...")
    top_100_ids = set(top_tier['candidate_id'].tolist())
    raw_candidate_map = {}

    # Fast forward through the raw JSONL file, only keeping our top 100
    with open(candidates_file, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            if cand.get("candidate_id") in top_100_ids:
                raw_candidate_map[cand["candidate_id"]] = cand
                if len(raw_candidate_map) == 100:
                    break  # Stop reading once we found all 100 winners

    # Apply the honest reasoning generator
    top_tier['reasoning'] = top_tier.apply(
        lambda row: generate_honest_reasoning(raw_candidate_map.get(row['candidate_id'], {}), row['rank']),
        axis=1
    )

    final_output = top_tier[['candidate_id', 'rank', 'rounded_final', 'reasoning']].rename(
        columns={'rounded_final': 'score'}
    )

    print(f">> Writing payload to disk [{output_csv}]...")
    final_output.to_csv(output_csv, index=False)

    t1 = time.time()
    print(f"Engine Shutdown. Cycle completed in {t1 - t0:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob Hackathon.")
    parser.add_argument('--candidates', type=str, required=True, help="Path to candidates.jsonl")
    parser.add_argument('--out', type=str, required=True, help="Path to output submission CSV")
    args = parser.parse_args()

    execute_scoring_pipeline(args.candidates, args.out)
