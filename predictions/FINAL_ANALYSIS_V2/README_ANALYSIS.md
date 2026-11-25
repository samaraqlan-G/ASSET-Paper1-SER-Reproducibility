# JOIV Final Post-hoc Analysis Package

Analysis engine version: 2.0.0

## Frozen input
Experiment root: `/content/drive/MyDrive/project_thesis_final/SER_EXPERIMENTS_V3`

Expected prediction matrix:
- 8 experiments
- 3 seeds (42,52,62)
- 5 frozen evaluation sets
- Total = 120 prediction files

Verified input files: 120

## No training or model selection
This package reads only the already frozen E1-E8 per-utterance predictions.
It does not train, fine-tune, refreeze, or select any model.

## Exact metrics
For every experiment/seed/corpus:
- Macro Precision
- Accuracy
- UAR (six-class macro recall)
- Macro-F1
- MCC
- Cohen's kappa

Mean ± SD is calculated over the three training seeds.

## Pairwise inferential analysis
Primary proposed-system contrasts:
E3 versus E1, E2, E4, E5, E6, E7, and E8.

For UAR and Macro-F1:
- 5,000 class-stratified paired bootstrap resamples
- 5,000 paired prediction-swap permutations
- same frozen utterance is kept paired across models and seeds
- Holm family-wise correction within each corpus and metric

The bootstrap/permutation procedures quantify frozen-test-sample uncertainty
conditional on the three trained models. Seed-to-seed SD is reported separately.
With only three training seeds, do not make broad population-level claims about
training-run significance.

For accuracy:
- exact paired McNemar test per seed
- Holm correction across the seven E3 comparator contrasts per corpus/seed
- per-seed tests are NOT treated as independent replications because the same
  frozen test utterances are reused.

## Cross-language interpretation
MALEO is the same-language Indonesian target. RAVDESS and CREMA-D are English,
and EMO-DB is German. The same-vs-cross-language comparison is descriptive,
not causal, because language is confounded with corpus, speakers, recording
conditions, acting style, annotation practices, and class distributions.

## JOIV outputs
The journal states that manuscripts should use the JOIV template and that
originality, clarity of presentation, and scientific significance are considered
during review. This package therefore emphasizes reproducible tables, paired
statistics, publication-quality figures, and conservative claim wording.

Generated output root:
`/content/drive/MyDrive/project_thesis_final/SER_EXPERIMENTS_V3/JOIV_FINAL_ANALYSIS_V2`
