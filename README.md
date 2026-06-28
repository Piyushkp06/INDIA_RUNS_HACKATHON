<div align="center">

# Jobs Match AI
### Intelligent Candidate Discovery & Ranking System

**India Runs Data & AI Challenge 2026** | Redrob AI Hackathon

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-4285F4?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://jobsmatchai.streamlit.app/)
[![License](https://img.shields.io/badge/License-Hackathon-green?style=for-the-badge)](LICENSE)

---

*A production-grade, multi-stage semantic ranking pipeline that discovers and ranks the top 100 candidates for Redrob AI's Senior AI Engineer (Founding Team) role from a pool of 5,000+ candidate profiles.*

[🚀 Live Demo](https://jobsmatchai.streamlit.app/) · [📊 Sample Output](#-sample-output) · [🏗️ Architecture](#-pipeline-architecture) · [⚡ Quick Start](#-quick-start)

</div>

---

## 🎯 The Challenge

Given a JSONL dataset of ~100k candidate profiles with career histories, skills, education, and behavioral signals — **rank the top 100 candidates** for a Senior AI Engineer role on Redrob AI's founding team.

The system must:
- ✅ Run **fully offline** (no API calls during ranking)
- ✅ Complete within **5 minutes on CPU** with 16 GB RAM
- ✅ Detect and penalize **honeypot/synthetic profiles**
- ✅ Produce **evidence-based reasoning** for every ranked candidate

---

## 🏗️ Pipeline Architecture

```
                          ┌─────────────────────────────┐
                          │    candidates.jsonl (100K+) │
                          └──────────────┬──────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │         STAGE 0: PRECOMPUTE (Offline)   │
                    │   Parse profiles → Extract features     │
                    │   Encode with MiniLM → Build FAISS index│
                    │   ⚡ Kaggle 2×T4 GPU | ~5 min          │
                    └────────────────────┬────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                INFERENCE PIPELINE (rank.py)         │
              │                                                     │
              │  ┌─────────────────────────────────────────────┐    │
              │  │ STAGE 1: Dense Semantic Retrieval           │    │
              │  │ all-MiniLM-L6-v2 + FAISS cosine similarity  │    │
              │  │ Retrieves all candidates ranked by JD match │    │ 
              │  └──────────────────┬──────────────────────────┘    │
              │                     │                               │
              │  ┌──────────────────▼──────────────────────────┐    │
              │  │ STAGE 2: Hard Disqualifier & Trap Filtering │    │
              │  │ Honeypots · Blocked titles · Agency-only    │    │
              │  │ Job hoppers · Pure research · LangChain trap│    │
              │  └──────────────────┬──────────────────────────┘    │
              │                     │  Top 250                      │
              │  ┌──────────────────▼──────────────────────────┐    │
              │  │ STAGE 3: Cross-Encoder Reranking            │    │
              │  │ ms-marco-MiniLM-L-6-v2 deep attention       │    │
              │  │ Pairwise JD↔Candidate relevance scoring     │    │
              │  └──────────────────┬──────────────────────────┘    │
              │                     │                               │
              │  ┌──────────────────▼──────────────────────────┐    │
              │  │ STAGE 4: Reciprocal Rank Fusion (RRF)       │    │
              │  │ Fuses dense + cross-encoder dual rankings   │    │ 
              │  │ k=60 smoothing parameter                    │    │
              │  └──────────────────┬──────────────────────────┘    │
              │                     │                               │
              │  ┌──────────────────▼──────────────────────────┐    │
              │  │ STAGE 5: Heuristic Blending & Reasoning     │    │
              │  │ Experience curve · Availability decay       │    │
              │  │ GitHub boost · Recruiter-save signal        │    │
              │  │ Evidence-based reasoning per candidate      │    │
              │  └──────────────────┬──────────────────────────┘    │
              │                     │                               │
              └─────────────────────┼───────────────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  submission.csv   │
                          │  Top 100 ranked   │
                          │  with reasoning   │
                          └───────────────────┘
```

---

## 🧠 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Bi-encoder + Cross-encoder** | Bi-encoder is fast for initial retrieval; cross-encoder provides deep pairwise attention for precision at the top |
| **RRF over linear combination** | RRF is rank-based (not score-based), making it robust to different score distributions from the two models |
| **Multiplicative penalties** | Honeypots/traps get multiplied by 0.01 — they can never accidentally rank high regardless of semantic match |
| **Experience bell-curve** | JD says 5–9 years; we use a peaked weight function instead of a hard cutoff for graceful degradation |
| **Evidence-based reasoning** | Each ranking is justified by citing specific keywords, production signals, and availability from the raw profile |

---

## 🛡️ Honeypot & Trap Detection

The pipeline includes 6 trap detectors to identify synthetic/adversarial profiles:

| Trap | Detection Logic |
|------|----------------|
| **Unverified Experts** | ≥3 "expert" skills with 0 months duration |
| **YoE Inflation** | Claimed experience differs from calculated months by >5 years |
| **Skill Duration Fraud** | Max skill months exceeds career length + 24 months |
| **Spam Applicants** | >10 applications in 30 days but 0% interview completion |
| **Ghost Profiles** | >90% response rate but inactive >1 year |
| **Phantom Seniors** | >8 YoE with no GitHub and no LinkedIn connected |

---

## ⚡ Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Option A: Full Pipeline (Pre-computed artifacts included)

The repo ships with pre-computed FAISS index and metadata via **Git LFS**, so you can run ranking directly:

```bash
# Clone with LFS
git lfs install
git clone https://github.com/Piyushkp06/INDIA_RUNS_HACKATHON.git
cd INDIA_RUNS_HACKATHON

# Run ranking (fully offline, ~2 min on CPU)
python rank.py --candidates ./candidates.jsonl --out ./Thala_For_A_Reason.csv

# Validate
python Data/India_runs_data_and_ai_challenge/validate_submission.py ./Thala_For_A_Reason.csv
```

### Option B: From Scratch

```bash
# 1. Download models (requires internet, one-time)
python download_models.py

# 2. Pre-compute embeddings & FAISS index
python precompute.py --candidates ./Data/India_runs_data_and_ai_challenge/candidates.jsonl

# 3. Run ranking
python rank.py --candidates ./Data/India_runs_data_and_ai_challenge/candidates.jsonl --out ./Thala_For_A_Reason.csv
```

---

## 📊 Sample Output

| Rank | Candidate | Score | Reasoning |
|------|-----------|-------|-----------|
| 1 | `CAND_0002025` | 0.9988 | 5.9yr Senior AI Engineer at Apple; strong match on weaviate, pinecone, opensearch; production evidence (production, deployed); actively looking, 80% response rate, 30d notice |
| 2 | `CAND_0068351` | 0.9421 | 6.4yr Lead AI Engineer at Sarvam AI; strong match on elasticsearch, qdrant; production evidence (production); passive, 86% response rate |
| 3 | `CAND_0081846` | 0.9144 | 6.7yr Lead AI Engineer at Razorpay; strong match on elasticsearch, qdrant, ndcg; production evidence (serving); actively looking, 73% response rate, 30d notice |

---

## 💻 Compute Environment

| Component | Specification |
|-----------|--------------|
| **Pre-computation** | Kaggle · 2× NVIDIA T4 GPU · ~5 minutes |
| **Inference (ranking)** | CPU only · No GPU required |
| **RAM** | 8–16 GB recommended |
| **Network** | ❌ Not required during ranking |
| **Runtime** | ~2 minutes for 100K candidates on CPU |

---

## 📁 Repository Structure

```
INDIA_RUNS_HACKATHON/
│
├── 🚀 rank.py                     # Main inference pipeline (5-stage ranker)
├── 🔧 precompute.py               # Offline embedding & FAISS index builder
├── 📥 download_models.py          # One-time model download utility
├── 🌐 app.py                      # Streamlit sandbox demo for evaluators
│
├── 🤖 model/                      # Saved bi-encoder (all-MiniLM-L6-v2)
├── 🤖 local_cross_encoder/        # Saved cross-encoder (ms-marco-MiniLM)
├── 📦 applicant_metadata.parquet  # Pre-computed candidate features
├── 📦 applicant_space.faiss       # Pre-computed FAISS vector index
│
├── 📄 requirements.txt            # Python dependencies
├── 📄 submission_metadata.yaml    # Hackathon metadata & declarations
├── 📄 Thala_For_A_Reason.csv      # Final submission (top 100 ranked)
├── 📓 Thala_For_A_Reason_Demo.ipynb  # Google Colab demo notebook
│
└── 📂 Data/
    └── India_runs_data_and_ai_challenge/
        ├── candidates.jsonl        # Full dataset (~487 MB, gitignored)
        ├── candidate_schema.json   # JSON schema for candidate profiles
        ├── sample_submission.csv   # Reference submission format
        ├── validate_submission.py  # Official submission validator
        └── ...                     # Challenge docs (JD, spec, signals)
```

---

## 🔬 Models Used

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Bi-Encoder | 87 MB | Dense semantic retrieval via FAISS |
| [`ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Cross-Encoder | 87 MB | Pairwise reranking with deep attention |

Both models are saved locally and loaded offline — **zero network calls during ranking**.

---

## 🏗️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FAISS-4285F4?style=flat-square&logo=meta&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle" />
</p>

---

<div align="center">

**Built with ❤️ by Piyushkanta Panda**

*Thala For A Reason* 🏏

</div>
