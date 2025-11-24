#!/usr/bin/env python
import os
import sys
import argparse
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from MLstatkit.stats import Delong_test
import umap
import pingouin as pg

from radMLBench import loadData
from harmonization import combat_robust, bart_harmonize, riemannian_harmonization
from preprocessing.nan_handling import handle_nans

warnings.filterwarnings('ignore')
np.random.seed(0)
sns.set(style='whitegrid')

def make_logreg():
    return LogisticRegression(max_iter=3000, C=1.0, solver='lbfgs', random_state=0)

def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95, random_state=0):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            scores.append(roc_auc_score(y_b, s_b))
        except Exception:
            continue
    scores = np.array(scores)
    if scores.size == 0:
        return np.nan, np.nan, np.nan
    alpha = 1.0 - ci
    lower = np.percentile(scores, 100 * (alpha / 2.0))
    upper = np.percentile(scores, 100 * (1.0 - alpha / 2.0))
    return float(scores.mean()), float(lower), float(upper)

def batch_leakage_score(X, batches, n_bootstrap=1000, random_state=0):
    batch_aucs = []
    batch_scores = {}
    for b in np.unique(batches):
        yb = (batches == b).astype(int)
        n_pos = yb.sum()
        n_neg = len(yb) - n_pos
        min_class = min(n_pos, n_neg)
        if min_class < 2:
            batch_aucs.append(np.nan)
            batch_scores[b] = np.full_like(yb, 0.5, dtype=float)
            continue
        n_splits_b = max(2, min(5, int(min_class)))
        cv = StratifiedKFold(n_splits=n_splits_b, shuffle=True, random_state=0)
        clf = make_logreg()
        y_score_b = cross_val_predict(clf, X, yb, cv=cv, method='predict_proba')[:, 1]
        if len(np.unique(yb)) < 2:
            auc_b = np.nan
        else:
            auc_b = roc_auc_score(yb, y_score_b)
        batch_aucs.append(auc_b)
        batch_scores[b] = y_score_b
    batch_aucs = np.array(batch_aucs, dtype=float)
    mean_auc = float(np.nanmean(batch_aucs))
    valid_aucs = batch_aucs[~np.isnan(batch_aucs)]
    if valid_aucs.size > 1:
        rng = np.random.RandomState(random_state)
        boot_means = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(valid_aucs), len(valid_aucs))
            boot_means.append(np.mean(valid_aucs[idx]))
        boot_means = np.array(boot_means)
        alpha = 1.0 - 0.95
        ci_low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
        ci_high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    else:
        ci_low, ci_high = np.nan, np.nan
    return mean_auc, ci_low, ci_high, batch_scores

def bio_auc_with_ci(X, y, n_splits=5, n_bootstrap=1000, random_state=0):
    if len(np.unique(y)) < 2 or n_splits < 2:
        return np.nan, np.nan, np.nan, np.full_like(y, 0.5, dtype=float)
    clf = make_logreg()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_score = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    auc_point = roc_auc_score(y, y_score)
    _, ci_low, ci_high = bootstrap_auc_ci(y_true=y, y_score=y_score, n_bootstrap=n_bootstrap, ci=0.95, random_state=random_state)
    return auc_point, ci_low, ci_high, y_score

def bio_preservation_scores(X, y, n_splits=5, n_bootstrap=1000):
    bio_auc, bio_ci_low, bio_ci_high, y_score = bio_auc_with_ci(X, y, n_splits=n_splits, n_bootstrap=n_bootstrap, random_state=0)
    pvals = []
    for j in range(X.shape):
        try:
            _, p = ttest_ind(X[y == 0, j], X[y == 1, j], equal_var=False)
            pvals.append(p)
        except Exception:
            pvals.append(np.nan)
    n_sig = int(np.sum(np.array(pvals) < 0.05))
    return bio_auc, n_sig, (bio_ci_low, bio_ci_high), y_score

def icc_batch_effect(X, subject_ids, batch_ids):
    icc_values = []
    n_features = X.shape
    for j in range(n_features):
        df = pd.DataFrame({'subject': subject_ids, 'batch': batch_ids, 'rating': X[:, j]})
        df_unique = df.groupby(['subject', 'batch']).mean().reset_index()
        try:
            icc_result = pg.intraclass_corr(data=df_unique, targets='subject', raters='batch', ratings='rating')
            icc_value = icc_result[icc_result['Type'] == 'ICC2']['ICC'].values
            icc_values.append(icc_value)
        except Exception:
            icc_values.append(np.nan)
    icc_values = np.array(icc_values, dtype=float)
    if np.all(np.isnan(icc_values)):
        return np.nan, icc_values.tolist()
    return float(np.nanmean(icc_values)), icc_values.tolist()

def plot_umap_visualization(methods_dict, batches, y, subset_size=300, save_path=None):
    """UMAP visualization для всех методов"""
    n_samples = len(y)
    if n_samples > subset_size:
        indices = np.random.choice(n_samples, size=subset_size, replace=False)
    else:
        indices = np.arange(n_samples)
    
    batches_sub = batches[indices]
    y_sub = y[indices]
    
    n_methods = len(methods_dict)
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 4*n_methods))
    
    if n_methods == 1:
        axes = axes[np.newaxis, :]
    
    print(f"\\nComputing UMAP for {n_methods} methods ({len(indices)} samples)...")
    
    for i, (name, X_method) in enumerate(tqdm(methods_dict.items(), desc="UMAP")):
        X_sub = X_method[indices]
        
        reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.1)
        umap_emb = reducer.fit_transform(X_sub)
        
        # Batch effect
        scatter1 = axes[i, 0].scatter(umap_emb[:, 0], umap_emb[:, 1], c=batches_sub, 
                                      cmap='tab10', s=15, alpha=0.7, edgecolors='none')
        axes[i, 0].set_title(f'{name} - Batch Effect', fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('UMAP 1')
        axes[i, 0].set_ylabel('UMAP 2')
        cbar1 = plt.colorbar(scatter1, ax=axes[i, 0])
        cbar1.set_label('Batch ID', rotation=270, labelpad=15)
        
        # Biological signal
        scatter2 = axes[i, 1].scatter(umap_emb[:, 0], umap_emb[:, 1], c=y_sub, 
                                      cmap='viridis', s=15, alpha=0.7, edgecolors='none')
        axes[i, 1].set_title(f'{name} - Biological Signal', fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('UMAP 1')
        axes[i, 1].set_ylabel('UMAP 2')
        cbar2 = plt.colorbar(scatter2, ax=axes[i, 1])
        cbar2.set_label('Target', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\\n UMAP saved: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run harmonization on any radMLBench dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name from radMLBench')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--n-batches', type=int, default=4, help='Number of pseudo-batches')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Bootstrap iterations')
    parser.add_argument('--subset-size', type=int, default=300, help='UMAP subset size')
    
    args = parser.parse_args()
    
    # Определяем output directory
    if args.output_dir is None:
        output_dir = Path('results') / args.dataset.lower().replace('-', '_')
    else:
        output_dir = Path(args.output_dir)
    
    stats_dir = output_dir / 'stats'
    plots_dir = output_dir / 'plots'
    
    # Создаем директории
    stats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"RADIOMICS HARMONIZATION: {args.dataset}")
    print("="*80)
    
    print(f"\\nLoading dataset: {args.dataset}...")
    try:
        data = loadData(args.dataset, return_X_y=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    print(f"Data shape: {data.shape}")
    
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    data.to_csv(datasets_dir / f"{args.dataset}.csv", index=False)
    print(f"Dataset saved: datasets/{args.dataset}.csv")
    
    batch_keywords = ['location', 'center', 'site', 'institution', 'scanner', 
                      'batch', 'source', 'acquisition', 'vendor']
    batch_cols = [col for col in data.columns 
                  if any(keyword in col.lower() for keyword in batch_keywords)]
    
    feature_cols = [col for col in data.columns if col not in ['ID', 'Target'] + batch_cols]
    X = data[feature_cols].values
    y = data['Target'].values.astype(int)
    subject_ids = data['ID'].values
    
    if batch_cols:
        print(f"Batch columns found: {batch_cols}")
        if all(any(prefix in col for prefix in ['Location', 'Center', 'Site']) for col in batch_cols):
            batch_onehot = data[batch_cols].values
            batch_ids = np.argmax(batch_onehot, axis=1)
        elif len(batch_cols) == 1:
            from sklearn.preprocessing import LabelEncoder
            batch_ids = LabelEncoder().fit_transform(data[batch_cols])
        else:
            batch_combined = data[batch_cols].astype(str).agg('_'.join, axis=1)
            from sklearn.preprocessing import LabelEncoder
            batch_ids = LabelEncoder().fit_transform(batch_combined)
    else:
        print("  No batch columns found. Creating pseudo-batches via k-means...")
        pca_temp = PCA(n_components=50, random_state=0)
        X_pca_temp = pca_temp.fit_transform(StandardScaler().fit_transform(X))
        kmeans = KMeans(n_clusters=args.n_batches, random_state=0, n_init=10)
        batch_ids = kmeans.fit_predict(X_pca_temp)
        print(f"✓ Created {args.n_batches} pseudo-batches")
    
    print(f"\\nData loaded: X={X.shape}, y={y.shape}, batches={len(np.unique(batch_ids))}")
    print(f"Class balance: {np.bincount(y) / len(y)}")
    
    # ========== ПРЕДОБРАБОТКА ==========
    print("\\n" + "="*80)
    print("PREPROCESSING")
    print("="*80)
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    print("\\n1. Removing low-variance features...")
    var_threshold = VarianceThreshold(threshold=0.01)
    X_filtered = var_threshold.fit_transform(X_std)
    print(f"   Removed {(~var_threshold.get_support()).sum()} features")
    print(f"   Remaining: {X_filtered.shape} features")
    
    print("\\n2. Removing highly correlated features...")
    corr_matrix = np.corrcoef(X_filtered.T)
    upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr_pairs = np.where((np.abs(corr_matrix) > 0.95) & upper_tri)
    
    to_drop = set()
    for i, j in zip(high_corr_pairs, high_corr_pairs):
        if X_filtered[:, i].var() < X_filtered[:, j].var():
            to_drop.add(i)
        else:
            to_drop.add(j)
    
    feature_mask_corr = np.ones(X_filtered.shape, dtype=bool)
    feature_mask_corr[list(to_drop)] = False
    X_filtered = X_filtered[:, feature_mask_corr]
    
    print(f"   Removed {len(to_drop)} correlated features")
    print(f"   Final: {X_filtered.shape} features ({100*X_filtered.shape/X.shape:.1f}%)")
    
    X_std = X_filtered
    
    print("\\n" + "="*80)
    print("HARMONIZATION")
    print("="*80)
    
    GLOBAL_N_SPLITS = max(2, min(5, int(np.bincount(y).min())))
    
    print("\\nComBat...")
    t0 = time.time()
    X_combat_raw = combat_robust(X_std, batch_ids, covars=y.reshape(-1, 1))
    X_combat = handle_nans(X_combat_raw, method='impute', verbose=True)
    print(f"   Time: {time.time()-t0:.1f} sec")
    
    print("\\nBART/RF...")
    t0 = time.time()
    X_bart_raw = bart_harmonize(X_std, batch_ids, covars=y.reshape(-1, 1))
    X_bart = handle_nans(X_bart_raw, method='impute', verbose=True)
    print(f"   Time: {time.time()-t0:.1f} sec")
    
    print("\\nRiemannian...")
    t0 = time.time()
    X_rie_raw = riemannian_harmonization(X_std, batch_ids, covars=y.reshape(-1, 1), pca_k=60)
    X_rie = handle_nans(X_rie_raw, method='impute', verbose=True)
    print(f"   Time: {time.time()-t0:.1f} sec")
    
    methods = {
        'raw': X_std,
        'ComBat': X_combat,
        'BART/RF': X_bart,
        'Riemannian': X_rie
    }
    print("\\n" + "="*80)
    print("UMAP VISUALIZATION")
    print("="*80)
    
    plot_umap_visualization(
        methods,
        batch_ids,
        y,
        subset_size=args.subset_size,
        save_path=plots_dir / 'umap_visualization.png'
    )
    
    print("\\n" + "="*80)
    print("METRICS EVALUATION")
    print("="*80)
    
    results = {}
    y_scores_bio = {}
    batch_scores_per_method = {}
    
    for name, X_method in tqdm(methods.items(), desc="Methods"):
        print(f"\\n{name}:")
        print("  - Batch leakage...")
        batch_auc, batch_ci_low, batch_ci_high, batch_scores = batch_leakage_score(
            X_method, batch_ids, n_bootstrap=args.n_bootstrap, random_state=0
        )
        print("  - Bio-preservation...")
        bio_auc, n_sig, (bio_ci_low, bio_ci_high), y_score = bio_preservation_scores(
            X_method, y, n_splits=GLOBAL_N_SPLITS, n_bootstrap=args.n_bootstrap
        )
        print("  - ICC...")
        icc_score, icc_per_feature = icc_batch_effect(X_method, subject_ids, batch_ids)
        
        results[name] = {
            'batch_auc': batch_auc,
            'batch_ci_low': batch_ci_low,
            'batch_ci_high': batch_ci_high,
            'bio_auc': bio_auc,
            'bio_ci_low': bio_ci_low,
            'bio_ci_high': bio_ci_high,
            'n_significant': n_sig,
            'icc_score': icc_score
        }
        y_scores_bio[name] = y_score
        batch_scores_per_method[name] = batch_scores
    
    res_df = pd.DataFrame(results).T
    print("\\n Results:")
    print(res_df.round(4))
    print("\\n" + "="*80)
    print("DELONG TESTS")
    print("="*80)
    
    from itertools import combinations
    
    # DeLong для bio-AUC
    print("\\n DeLong test for bio-AUC:")
    delong_bio_rows = []
    method_names = list(methods.keys())
    
    for m1, m2 in combinations(method_names, 2):
        if np.allclose(y_scores_bio[m1], y_scores_bio[m1]) or np.allclose(y_scores_bio[m2], y_scores_bio[m2]):
            z_score, p_value = np.nan, np.nan
        else:
            try:
                delong_res = Delong_test(y, y_scores_bio[m1], y_scores_bio[m2])
                z_score, p_value = delong_res, delong_res
            except Exception:
                z_score, p_value = np.nan, np.nan
        
        delong_bio_rows.append({
            'model_1': m1,
            'model_2': m2,
            'auc_1': res_df.loc[m1, 'bio_auc'],
            'auc_2': res_df.loc[m2, 'bio_auc'],
            'z_score': z_score,
            'p_value': p_value,
            'significant': 'Yes' if (not np.isnan(p_value) and p_value < 0.05) else 'No'
        })
    
    delong_bio_df = pd.DataFrame(delong_bio_rows)
    print(delong_bio_df.to_string(index=False))
    
    # DeLong для batch AUC
    print("\\n DeLong test for batch AUC:")
    delong_batch_rows = []
    unique_batches = np.unique(batch_ids)
    
    for m1, m2 in combinations(method_names, 2):
        for b in unique_batches:
            yb = (batch_ids == b).astype(int)
            scores1 = batch_scores_per_method[m1].get(b, None)
            scores2 = batch_scores_per_method[m2].get(b, None)
            
            if scores1 is None or scores2 is None:
                continue
            if np.allclose(scores1, scores1) or np.allclose(scores2, scores2) or len(np.unique(yb)) < 2:
                continue
            
            try:
                delong_res = Delong_test(yb, scores1, scores2)
                z_score, p_value = delong_res, delong_res
                auc1 = roc_auc_score(yb, scores1)
                auc2 = roc_auc_score(yb, scores2)
            except Exception:
                continue
            
            delong_batch_rows.append({
                'batch': int(b),
                'model_1': m1,
                'model_2': m2,
                'auc_1': auc1,
                'auc_2': auc2,
                'z_score': z_score,
                'p_value': p_value,
                'significant': 'Yes' if (not np.isnan(p_value) and p_value < 0.05) else 'No'
            })
    
    delong_batch_df = pd.DataFrame(delong_batch_rows)
    print(delong_batch_df.head(10).to_string(index=False))
    print(f"\\n... ({len(delong_batch_df)} total comparisons)")
    print("\\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    res_df.to_csv(stats_dir / 'metrics.csv')
    delong_bio_df.to_csv(stats_dir / 'delong_bio.csv', index=False)
    delong_batch_df.to_csv(stats_dir / 'delong_batch.csv', index=False)
    
    print(f"\\n Results saved to:")
    print(f"   - {stats_dir / 'metrics.csv'}")
    print(f"   - {stats_dir / 'delong_bio.csv'}")
    print(f"   - {stats_dir / 'delong_batch.csv'}")
    print(f"   - {plots_dir / 'umap_visualization.png'}")
    print("\\n Done!")

if __name__ == '__main__':
    main()
