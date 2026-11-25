#!/usr/bin/env python3
"""ASSET reviewer-requested external no-attention control.

INFERENCE ONLY. Reuses the already-frozen B2 no-attention checkpoints from the
source-validation architecture-confirmation stage. No training, tuning,
selection, or checkpoint modification occurs.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

SEEDS=(42,52,62)
EVALS=('eseravd_internal_test','maleo','ravdess','cremad','emodb')

def load_module(path):
    spec=importlib.util.spec_from_file_location('canonical_v33',str(path)); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def metrics(y,p):
    return dict(accuracy=float(accuracy_score(y,p)),uar=float(balanced_accuracy_score(y,p)),macro_f1=float(f1_score(y,p,average='macro',zero_division=0)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-dir',default='/content/drive/MyDrive/project_thesis_final'); ap.add_argument('--base-script',default=None)
    args=ap.parse_args(); project=Path(args.project_dir)
    base=Path(args.base_script) if args.base_script else project/'code/ser_thesis_pipeline_v3_3_matched_domain_gap.py'
    if not base.exists():
        alt=project/'ser_thesis_pipeline_v3_3_matched_domain_gap.py'; base=alt if alt.exists() else base
    if not base.exists(): raise FileNotFoundError(f'Canonical v3.3 script not found: {base}')
    mod=load_module(base)
    pargs=mod.build_arg_parser().parse_args(['--mode','status','--project-dir',str(project),'--seeds','42,52,62','--skip-preflight'])
    # Mirror main() post-parsing normalization of comma-separated seeds if needed.
    if isinstance(pargs.seeds,str): pargs.seeds=[int(x) for x in pargs.seeds.split(',') if x.strip()]
    pipe=mod.SERThesisPipeline(pargs); pipe.setup()
    stage=project/'SER_EXPERIMENTS_V3/campaigns/stage_B3_bottleneck_confirm'
    cfg=json.loads((stage/'experiment_matrix.json').read_text())['B2_CNNLSTM_BOTTLENECK_CONFIRM']
    assert cfg['model']=='cnn_lstm_bottleneck' and cfg.get('attention') is False
    out=project/'SER_EXPERIMENTS_V3/reporting/asset_no_attention_external_v1'; out.mkdir(parents=True,exist_ok=True)
    frames={'eseravd_internal_test':pipe.internal_test_df, **pipe.external_frames}; rows=[]
    for seed in SEEDS:
        weights=stage/f'B2_CNNLSTM_BOTTLENECK_CONFIRM_seed{seed}'/'models'/'best.weights.h5'
        model=pipe._load_inference_model(weights,cfg)
        for ev in EVALS:
            seq=pipe.make_sequence(frames[ev],cfg,training=False,seed=seed,shuffle=False)
            y,p,prob=pipe.collect_predictions(model,seq); m=metrics(y,p); rows.append(dict(seed=seed,evaluation_set=ev,**m,n=len(y)))
            d=pd.DataFrame({'true_class_index':y,'pred_class_index':p});
            for j in range(prob.shape[1]): d[f'prob_class_{j+1}']=prob[:,j]
            d.to_csv(out/f'B2_no_attention_seed{seed}_{ev}_predictions.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(out/'B2_no_attention_metrics_by_seed.csv',index=False)
    agg=df.groupby('evaluation_set')[['accuracy','uar','macro_f1']].agg(['mean','std']); agg.to_csv(out/'B2_no_attention_metrics_mean_sd.csv')
    # Compare to the already frozen E3 predictions, per seed.
    comp=[]
    for seed in SEEDS:
        for ev in EVALS:
            b=df[(df.seed==seed)&(df.evaluation_set==ev)].iloc[0]
            ep=project/'SER_EXPERIMENTS_V3/campaigns/final_E1_E8'/f'E3_seed{seed}'/'results'/f'{ev}_predictions.csv'
            e=pd.read_csv(ep); em=metrics(e.true_class_index.to_numpy(int),e.pred_class_index.to_numpy(int))
            comp.append(dict(seed=seed,evaluation_set=ev,E3_uar=em['uar'],NoAttention_uar=b.uar,delta_uar=em['uar']-b.uar,E3_macro_f1=em['macro_f1'],NoAttention_macro_f1=b.macro_f1,delta_macro_f1=em['macro_f1']-b.macro_f1))
    c=pd.DataFrame(comp); c.to_csv(out/'E3_vs_no_attention_by_seed.csv',index=False)
    ca=c.groupby('evaluation_set')[['E3_uar','NoAttention_uar','delta_uar','E3_macro_f1','NoAttention_macro_f1','delta_macro_f1']].agg(['mean','std']); ca.to_csv(out/'E3_vs_no_attention_mean_sd.csv')
    (out/'README.md').write_text('INFERENCE ONLY: frozen B2 no-attention checkpoints evaluated on the same frozen internal/external frames as E3. No model was trained, tuned, selected, or modified.\n',encoding='utf-8')
    print(ca); print('\nWROTE',out)
if __name__=='__main__': main()
