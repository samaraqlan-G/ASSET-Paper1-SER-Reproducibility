# ASSET cluster-aware revision audit

Analysis only; no training/tuning/model selection.

Cluster fields: E-SERAVD internal = frozen source_group_code (semantics not relabelled); RAVDESS/CREMA-D/EMO-DB = canonical speaker_id. MALEO is not cluster-resampled because the frozen canonical metadata does not contain a defensible speaker/group identifier.

Three seeds enter each contrast as follows: the metric difference is computed separately for seeds 42, 52 and 62 and averaged. Every bootstrap cluster sample and every permutation cluster-swap pattern is shared across E3/comparator and all seeds. Seeds are therefore repeated trained-model realizations, not independent test observations.

E-SERAVD internal has only two held-out source_group_code clusters, so its cluster-aware interval/p-value is necessarily coarse and is reported as a sensitivity check rather than strong population inference.

Bootstrap replicates: 10000.
