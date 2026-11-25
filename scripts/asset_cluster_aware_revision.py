#!/usr/bin/env python3
"""ASSET Paper 1 - reviewer-requested cluster-aware sensitivity analysis.

ANALYSIS ONLY: reads frozen predictions and metadata. It does not train, tune,
select, refreeze, or modify any model/prediction.

Cluster definitions:
- E-SERAVD internal: frozen source_group_code (2 held-out groups; semantics are
  intentionally NOT relabelled as speakers or movie titles).
- RAVDESS / CREMA-D / EMO-DB: canonical speaker_id where verified/parsed.
- MALEO: no defensible speaker/group identifier in the frozen canonical metadata;
  cluster-aware inference is therefore not fabricated for MALEO.

The paired effect is the mean of the three seed-specific metric differences.
Cluster resampling and prediction swaps use the same sampled clusters/swaps for
E3 and comparator and for all three seeds, so seed predictions are never treated
as independent observations.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

SEEDS=(42,52,62)
COMPARATORS=("E1","E2","E4","E5","E6","E7","E8")
METRICS=("uar","macro_f1")
EVALS={
    "eseravd_internal_test": ("internal", "source_group_code"),
    "ravdess": ("ravdess", "speaker_id"),
    "cremad": ("cremad", "speaker_id"),
    "emodb": ("emodb", "speaker_id"),
}

def metric(y,p,name):
    if name=="uar": return float(balanced_accuracy_score(y,p))
    if name=="macro_f1": return float(f1_score(y,p,average="macro",zero_division=0))
    raise KeyError(name)

def holm(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]; running=max(running,val); adj[idx]=min(1.0,running)
    return adj

def metadata_frames(project):
    manifest=pd.read_csv(project/'DATA_CANONICAL_V2/manifests/analysis_manifest_final.csv')
    split=pd.read_csv(project/'SER_EXPERIMENTS_V2/splits/eseravd_group_disjoint_split_v1.csv')
    out={}
    s=split.loc[split['split'].astype(str).eq('test')].copy().reset_index(drop=True)
    s['class_index']=s['label_id'].astype(int)-1
    out['eseravd_internal_test']=s
    for ds in ('ravdess','cremad','emodb'):
        d=manifest.loc[manifest['dataset'].astype(str).str.lower().eq(ds)].copy().reset_index(drop=True)
        d['class_index']=d['label_id'].astype(int)-1
        if d['speaker_id'].isna().any():
            raise RuntimeError(f'{ds}: missing speaker_id in canonical manifest')
        out[ds]=d
    return out

def pred_path(project, exp, seed, eval_name):
    return project/'SER_EXPERIMENTS_V3/campaigns/final_E1_E8'/f'{exp}_seed{seed}'/'results'/f'{eval_name}_predictions.csv'

def load_preds(project, exp, eval_name, meta):
    out={}
    for seed in SEEDS:
        p=pred_path(project,exp,seed,eval_name)
        if not p.exists(): raise FileNotFoundError(p)
        df=pd.read_csv(p)
        if len(df)!=len(meta): raise RuntimeError(f'{p}: n={len(df)} != metadata n={len(meta)}')
        y=df['true_class_index'].to_numpy(int); pred=df['pred_class_index'].to_numpy(int)
        if not np.array_equal(y,meta['class_index'].to_numpy(int)):
            raise RuntimeError(f'{p}: frozen prediction row order/labels do not match canonical metadata')
        out[seed]=(y,pred)
    return out

def effect(predA,predB,idx,metric_name):
    vals=[]
    for seed in SEEDS:
        y,a=predA[seed]; _,b=predB[seed]
        vals.append(metric(y[idx],a[idx],metric_name)-metric(y[idx],b[idx],metric_name))
    return float(np.mean(vals))

def cluster_bootstrap(predA,predB,clusters,metric_name,reps,rng):
    uniq=np.array(pd.unique(clusters),dtype=object)
    members={g:np.flatnonzero(clusters==g) for g in uniq}
    draws=np.empty(reps,float)
    for r in range(reps):
        sampled=rng.choice(uniq,size=len(uniq),replace=True)
        idx=np.concatenate([members[g] for g in sampled])
        draws[r]=effect(predA,predB,idx,metric_name)
    return np.quantile(draws,[.025,.975]).tolist(), draws

def perm_effect(predA,predB,clusters,metric_name,swap):
    # swap whole cluster across every seed; preserve pairing.
    vals=[]
    uniq=np.array(pd.unique(clusters),dtype=object)
    swap_groups=set(uniq[np.asarray(swap,bool)])
    mask=np.array([g in swap_groups for g in clusters],bool)
    for seed in SEEDS:
        y,a=predA[seed]; _,b=predB[seed]
        ap=a.copy(); bp=b.copy(); ap[mask]=b[mask]; bp[mask]=a[mask]
        vals.append(metric(y,ap,metric_name)-metric(y,bp,metric_name))
    return float(np.mean(vals))

def cluster_permutation(predA,predB,clusters,metric_name,reps,rng):
    uniq=np.array(pd.unique(clusters),dtype=object); k=len(uniq)
    obs=effect(predA,predB,np.arange(len(clusters)),metric_name)
    if k<=12:
        vals=[]
        for bits in range(1<<k):
            swap=[bool((bits>>j)&1) for j in range(k)]
            vals.append(perm_effect(predA,predB,clusters,metric_name,swap))
        vals=np.asarray(vals)
        p=float(np.mean(np.abs(vals)>=abs(obs)-1e-15)); method=f'exact_{1<<k}'
    else:
        vals=np.empty(reps,float)
        for r in range(reps):
            vals[r]=perm_effect(predA,predB,clusters,metric_name,rng.random(k)<.5)
        p=float((1+np.sum(np.abs(vals)>=abs(obs)-1e-15))/(reps+1)); method=f'monte_carlo_{reps}'
    return obs,p,method

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-dir',default='/content/drive/MyDrive/project_thesis_final')
    ap.add_argument('--reps',type=int,default=10000); ap.add_argument('--seed',type=int,default=4696)
    args=ap.parse_args(); project=Path(args.project_dir); out=project/'SER_EXPERIMENTS_V3/reporting/asset_cluster_aware_revision_v1'; out.mkdir(parents=True,exist_ok=True)
    meta=metadata_frames(project); rng=np.random.default_rng(args.seed); rows=[]
    for eval_name,(dataset,cluster_col) in EVALS.items():
        md=meta[eval_name]; clusters=md[cluster_col].astype(str).to_numpy(); k=len(pd.unique(clusters))
        e3=load_preds(project,'E3',eval_name,md)
        for comp in COMPARATORS:
            b=load_preds(project,comp,eval_name,md)
            for m in METRICS:
                obs,p,pmethod=cluster_permutation(e3,b,clusters,m,args.reps,rng)
                ci,draws=cluster_bootstrap(e3,b,clusters,m,args.reps,rng)
                rows.append(dict(evaluation_set=eval_name,dataset=dataset,cluster_field=cluster_col,n_clusters=k,
                    contrast=f'E3-{comp}',metric=m,delta=obs,ci_low=ci[0],ci_high=ci[1],cluster_permutation_p=p,
                    permutation_method=pmethod,bootstrap_reps=args.reps,notes=('ONLY TWO HELD-OUT SOURCE GROUPS; coarse sensitivity' if eval_name=='eseravd_internal_test' else '')))
    df=pd.DataFrame(rows)
    df['holm_p']=np.nan
    for (ev,m),ix in df.groupby(['evaluation_set','metric']).groups.items():
        df.loc[ix,'holm_p']=holm(df.loc[ix,'cluster_permutation_p'].to_numpy(float))
    df['holm_significant_0_05']=df['holm_p']<.05
    df['cluster_ci_excludes_zero']=(df['ci_low']>0)|(df['ci_high']<0)
    df.to_csv(out/'ASSET_cluster_aware_all_contrasts.csv',index=False)
    summary=df[['evaluation_set','n_clusters','contrast','metric','delta','ci_low','ci_high','holm_p','holm_significant_0_05','cluster_ci_excludes_zero','notes']]
    summary.to_csv(out/'ASSET_cluster_aware_reviewer_table.csv',index=False)
    readme=f'''# ASSET cluster-aware revision audit\n\nAnalysis only; no training/tuning/model selection.\n\nCluster fields: E-SERAVD internal = frozen source_group_code (semantics not relabelled); RAVDESS/CREMA-D/EMO-DB = canonical speaker_id. MALEO is not cluster-resampled because the frozen canonical metadata does not contain a defensible speaker/group identifier.\n\nThree seeds enter each contrast as follows: the metric difference is computed separately for seeds 42, 52 and 62 and averaged. Every bootstrap cluster sample and every permutation cluster-swap pattern is shared across E3/comparator and all seeds. Seeds are therefore repeated trained-model realizations, not independent test observations.\n\nE-SERAVD internal has only two held-out source_group_code clusters, so its cluster-aware interval/p-value is necessarily coarse and is reported as a sensitivity check rather than strong population inference.\n\nBootstrap replicates: {args.reps}.\n'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    print(summary.to_string(index=False)); print('\nWROTE',out)
if __name__=='__main__': main()
