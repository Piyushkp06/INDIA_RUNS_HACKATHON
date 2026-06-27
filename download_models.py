"""
download_models.py
------------------
Downloads and saves the two transformer models needed for offline ranking.
Run this ONCE on a machine with internet access. The saved model directories
are then bundled with the repo so that `rank.py` runs fully offline.

Usage:
    python download_models.py
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import os

BIENCODER_NAME = "all-MiniLM-L6-v2"
CROSSENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BIENCODER_DIR = os.path.join(os.path.dirname(__file__), "model")
CROSSENCODER_DIR = os.path.join(os.path.dirname(__file__), "local_cross_encoder")


def main():
    print(f"[1/2] Downloading bi-encoder: {BIENCODER_NAME} ...")
    bi_model = SentenceTransformer(BIENCODER_NAME)
    bi_model.save(BIENCODER_DIR)
    print(f"       Saved to {BIENCODER_DIR}")

    print(f"[2/2] Downloading cross-encoder: {CROSSENCODER_NAME} ...")
    cross_model = CrossEncoder(CROSSENCODER_NAME)
    cross_model.save(CROSSENCODER_DIR)
    print(f"       Saved to {CROSSENCODER_DIR}")

    print("\nDone! Both models are ready for offline use.")


if __name__ == "__main__":
    main()
