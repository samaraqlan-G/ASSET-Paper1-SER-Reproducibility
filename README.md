# ASSET Paper 1 Reproducibility Package v1

Purpose: reviewer-facing reproduction of the frozen source split, cluster-aware statistical revision, matched no-attention external control, and final E3-vs-HuBERT/no-attention cluster-aware closure.

Scientific guardrails:
- No target corpus was used for model selection.
- R4A and R4C are analysis-only.
- R4B uses already-frozen B2 no-attention checkpoints; no retraining was performed.
- E-SERAVD internal cluster unit: source_group_code (semantics not relabelled as speaker/movie).
- RAVDESS, CREMA-D, EMO-DB cluster unit: speaker_id.
- MALEO pairwise inference is descriptive because no defensible repeated speaker/group identifier exists in the frozen metadata.
- Significance uses cluster-level paired permutation with Holm correction; bootstrap intervals support effect-size uncertainty.

# Repository Root Structure
ASSET-Paper1-SER-Reproducibility/
├── README.md
├── SHA256_MANIFEST.csv
├── metadata/
├── notebooks/
├── predictions/
├── results/
├── scripts/
└── splits/
