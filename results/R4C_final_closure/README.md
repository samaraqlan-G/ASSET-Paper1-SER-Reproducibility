# ASSET Final Cluster-Aware Closure v1

Analysis only. No model was trained, tuned, selected, refrozen, or modified.

Contrasts:
- E3 vs frozen B2 no-attention
- E3 vs frozen S2 HuBERT

Cluster units:
- E-SERAVD internal: source_group_code
- RAVDESS / CREMA-D / EMO-DB: speaker_id
- MALEO: descriptive only because no defensible repeated speaker/group identifier is available.

Significance rule:
Holm-adjusted cluster-level paired permutation p < 0.05.
Bootstrap confidence intervals are supporting effect-size uncertainty, not the primary significance decision.

E-SERAVD internal has only two held-out source groups and is therefore a coarse sensitivity analysis.
