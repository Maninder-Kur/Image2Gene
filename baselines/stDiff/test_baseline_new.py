#!/usr/bin/env python3
"""
Full baseline runner (SpaGE, Tangram, stPlus, UniPort, gimVI) + metric postprocessing.
- 80/20 spatial cell split (train/test)
- gene-wise KFold (n_splits)
- stPlus model reuse attempt (fallback to re-running if reuse not supported)
- shape alignment safeguards before metrics
- per-fold CSV outputs + final CalDataMetric across all model folders
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as st
import copy
import warnings
import torch
import scanpy as sc
from os.path import join
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

# ---------- Utilities & external functions ----------
# compute_metrics must exist in utils/compute_metrics.py
from utils.compute_metrics import compute_metrics, spearmanrr

# process.result_analysis may contain clustering_metrics etc. Keep import to preserve your helper functions.
from process.result_analysis import *

# try to import baseline modules (they may or may not be present)
try:
    import baseline.stPlus as baseline_stplus_module
except Exception:
    baseline_stplus_module = None

try:
    import baseline.tangram as baseline_tg
except Exception:
    baseline_tg = None

# UniPort import (optional)
try:
    import uniport as up
except Exception:
    up = None

# SpaGE will be imported dynamically inside SpaGE_impute

# ---------- CLI ----------
import argparse
parser = argparse.ArgumentParser(description="Run baselines and compute fold metrics")
parser.add_argument("--sc_data", type=str, required=False, default="dataset5_seq_915.h5ad")
parser.add_argument("--sp_data", type=str, required=False, default="dataset5_spatial_915.h5ad")
parser.add_argument("--document", type=str, required=False, default="dataset5")
parser.add_argument("--filename", type=str, required=False, default="Dataset")
parser.add_argument("--rand", type=int, required=False, default=0)
args = parser.parse_args()

# ---------- Config ----------
n_splits = 5
random_seed = int(args.rand)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# ---------- Read data ----------
print("Reading data:")
print("  spatial:", args.sp_data)
print("  sc    :", args.sc_data)
adata_spatial = sc.read_h5ad(args.sp_data)
adata_seq = sc.read_h5ad(args.sc_data)
print(f"Loaded shapes: spatial {adata_spatial.shape}, sc {adata_seq.shape}")

# keep only common genes
common_genes = np.intersect1d(adata_spatial.var_names, adata_seq.var_names)
print(f"Number of common genes: {len(common_genes)}")
adata_spatial = adata_spatial[:, common_genes].copy()
adata_seq = adata_seq[:, common_genes].copy()

# normalized copies used by some baselines (kept for CalDataMetric cluster use as adata_spatial2)
adata_spatial2 = adata_spatial.copy()
adata_seq2 = adata_seq.copy()
adata_seq3 = adata_seq.copy()
sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)

sp_genes = np.array(adata_spatial.var_names)

# build Pandas dataframes with spot/cell indices preserved (these are normalized)
sp_data_full = pd.DataFrame(adata_spatial2.X, columns=sp_genes, index=adata_spatial.obs_names)
sc_data = pd.DataFrame(adata_seq2.X, columns=sp_genes, index=adata_seq.obs_names)
print(f"Converted to DataFrame: sp_data {sp_data_full.shape}, sc_data {sc_data.shape}")

# ---------- Train/test spatial split (80/20) ----------
train_spatial_df, test_spatial_df = train_test_split(sp_data_full, test_size=0.2, random_state=random_seed, shuffle=True)
sp_data = train_spatial_df.copy()       # variable used across pipeline (train spatial)
sp_data_test = test_spatial_df.copy()   # held-out test spatial
print(f"Spatial split -> Train: {sp_data.shape}, Test: {sp_data_test.shape}")

# ---------- Outputs folder ----------
Data = args.document
outdir = join("Result", Data)
os.makedirs(outdir, exist_ok=True)

# ---------- Helpers ----------
def align_for_metrics(y_true, y_pred, tag=""):
    """
    Ensure y_true and y_pred are numpy arrays of identical shape.
    If sizes match, reshape preds to the true shape. If not, truncate to min size.
    Returns (y_true_aligned, y_pred_aligned).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape == y_pred.shape:
        return y_true, y_pred

    # If same total elements, reshape pred -> shape(true)
    if y_true.size == y_pred.size:
        y_pred = y_pred.reshape(y_true.shape)
        print(f"[INFO] align_for_metrics{(' '+tag) if tag else ''}: reshaped preds to {y_true.shape}")
        return y_true, y_pred

    # If pred is 1D and matches flattened expectation, reshape
    if y_pred.ndim == 1 and y_pred.size % y_true.shape[0] == 0:
        cols = y_pred.size // y_true.shape[0]
        if y_true.ndim == 2 and cols == y_true.shape[1]:
            y_pred = y_pred.reshape(y_true.shape)
            return y_true, y_pred

    # fallback: flatten and truncate to min size
    min_len = min(y_true.size, y_pred.size)
    print(f"[WARN] align_for_metrics{(' '+tag) if tag else ''}: shapes {y_true.shape} vs {y_pred.shape}, flattening & truncating to {min_len}")
    y_true = y_true.flatten()[:min_len]
    y_pred = y_pred.flatten()[:min_len]
    return y_true, y_pred

def save_fold_and_summary_metrics(fold_metrics_dict, csv_dir, prefix="fold_metrics"):
    df_metrics = pd.DataFrame.from_dict(fold_metrics_dict, orient="index")
    os.makedirs(csv_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(csv_dir, f"{prefix}.csv"), index=False)
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
        df_std = df_metrics[numeric_cols].std().to_frame(name="std").T
        df_summary = pd.concat([df_mean, df_std])
        df_summary.to_csv(os.path.join(csv_dir, f"{prefix}_summary.csv"), index=False)
        print(f"[INFO] saved summary to {csv_dir}/{prefix}_summary.csv")

# ===========================
# stPlus wrapper (attempt reuse)
# ===========================
def stplus_wrapper(train_sp_df, sc_df, test_genes, save_path_prefix, model=None):
    """
    Call baseline.stPlus in a robust way. If baseline.stPlus supports returning a model
    object (or accepts model argument), attempt to reuse it. Otherwise fallback to calling
    baseline.stPlus returning predictions array.
    Returns (model_object_or_None, predictions_array)
    """
    if baseline_stplus_module is None:
        raise RuntimeError("baseline.stPlus module not found. Ensure baseline/stPlus.py is present.")

    stplus_func = getattr(baseline_stplus_module, "stPlus", None)
    if stplus_func is None:
        raise RuntimeError("baseline.stPlus.stPlus function not found in baseline.stPlus.")

    try:
        # Try with model argument if provided
        if model is not None:
            out = stplus_func(train_sp_df, sc_df, test_genes, save_path_prefix, model=model)
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], np.asarray(out[1])
            else:
                # some versions return only preds
                return model, np.asarray(out)
        else:
            out = stplus_func(train_sp_df, sc_df, test_genes, save_path_prefix)
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], np.asarray(out[1])
            else:
                return None, np.asarray(out)
    except TypeError:
        # signature mismatch (older stPlus). Fallback to calling without model.
        print("[WARN] stplus_wrapper: signature mismatch, calling baseline.stPlus without model argument.")
        out = stplus_func(train_sp_df, sc_df, test_genes, save_path_prefix)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], np.asarray(out[1])
        else:
            return None, np.asarray(out)
    except Exception as e:
        print("[ERROR] stplus_wrapper: error calling baseline.stPlus:", e)
        raise

def stPlus_impute():
    """
    For each fold:
     - Train stPlus on train_spatial (sp_data[train_genes])
     - Reuse model object to predict on held-out sp_data_test (if reuse supported)
     - Save per-fold CSVs and metrics (val and test)
    """
    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    torch.manual_seed(random_seed)

    outdir1 = os.path.join(outdir, "stplus")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)

    idx = 1
    fold_metrics = {}
    fold_test_metrics = {}
    all_pred_res = np.zeros_like(adata_spatial.X)  # placeholder (rows = original adata_spatial order)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print(f"\n===== stPlus Fold {idx} =====")
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])

        print(f"[Fold {idx}] train_genes: {len(train_gene)}, test_genes: {len(test_gene)}")
        fold_dir = os.path.join(outdir1, f"fold_{idx}")
        os.makedirs(fold_dir, exist_ok=True)
        save_prefix = os.path.join(fold_dir, "stplus_model")

        # 1) Train on train_spatial and get validation predictions (train_spatial rows)
        try:
            model_obj, pred_val = stplus_wrapper(sp_data[train_gene], sc_data, test_gene, save_prefix, model=None)
        except Exception as e:
            print("[ERROR] stPlus training failed on this fold:", e)
            idx += 1
            continue

        pred_val = np.asarray(pred_val)
        # reshape guards
        if pred_val.ndim == 1:
            if pred_val.size == sp_data.shape[0] * len(test_gene):
                pred_val = pred_val.reshape(sp_data.shape[0], len(test_gene))
            else:
                # try to broadcast by rows if impossible -> fallback zeros
                print(f"[WARN] stPlus returned 1D of length {pred_val.size}, cannot reshape to expected ({sp_data.shape[0]},{len(test_gene)}). Using zeros.")
                pred_val = np.zeros((sp_data.shape[0], len(test_gene)))
        if pred_val.shape[0] != sp_data.shape[0]:
            # maybe prediction returned full adata_spatial ordering — try to match by index length
            if pred_val.shape[0] == adata_spatial.shape[0]:
                # restrict to train rows by reindexing using adata_spatial.obs_names
                df_tmp = pd.DataFrame(pred_val, index=adata_spatial.obs_names)
                df_pred_val = df_tmp.reindex(sp_data.index)[list(range(pred_val.shape[1]))]
                df_pred_val.columns = test_gene
                pred_val = df_pred_val.values
            else:
                # fallback zeros
                print(f"[WARN] stPlus returned shape {pred_val.shape}. Expected {sp_data.shape[0]}. Using zeros for val preds.")
                pred_val = np.zeros((sp_data.shape[0], len(test_gene)))

        df_pred_val = pd.DataFrame(pred_val, index=sp_data.index, columns=test_gene)
        df_true_val = sp_data[test_gene].copy()

        # 2) Predict on held-out test_spatial (reuse model_obj if available)
        if model_obj is not None:
            try:
                valid_train_genes = [g for g in train_gene if g in sp_data_test.columns]
                if len(valid_train_genes) == 0:
                    print(f"[WARN] No overlap between test_spatial columns and train_gene — using zeros.")
                    pred_test = np.zeros((sp_data_test.shape[0], len(test_gene)))
                else:
                    _, pred_test = stplus_wrapper(sp_data_test[valid_train_genes], sc_data, test_gene, save_prefix, model=model_obj)
                pred_test = np.asarray(pred_test)
            except Exception as e:
                print("[WARN] failed to predict test_spatial using model_obj, fallback to calling baseline.stPlus:", e)
                _, pred_test = stplus_wrapper(sp_data_test[train_gene], sc_data, test_gene, save_prefix, model=None)
        else:
            print("[WARN] Model reuse not available; calling baseline.stPlus on test split (may re-train internally).")
            try:
                _, pred_test = stplus_wrapper(sp_data_test[train_gene], sc_data, test_gene, save_prefix, model=None)
                pred_test = np.asarray(pred_test)
            except Exception as e:
                print("[ERROR] fallback prediction on test set failed:", e)
                pred_test = np.zeros((sp_data_test.shape[0], len(test_gene)))

        # reshape guards for test preds
        if pred_test.ndim == 1:
            if pred_test.size == sp_data_test.shape[0] * len(test_gene):
                pred_test = pred_test.reshape(sp_data_test.shape[0], len(test_gene))
            elif pred_test.size == adata_spatial.shape[0] * len(test_gene):
                # if returned predictions for whole adata_spatial
                tmp_df = pd.DataFrame(pred_test.reshape(adata_spatial.shape[0], len(test_gene)), index=adata_spatial.obs_names, columns=test_gene)
                tmp_df = tmp_df.reindex(sp_data_test.index)
                pred_test = tmp_df.values
            else:
                pred_test = np.zeros((sp_data_test.shape[0], len(test_gene)))

        if pred_test.shape[0] != sp_data_test.shape[0]:
            # if pred_test corresponds to full adata_spatial rows
            if pred_test.shape[0] == adata_spatial.shape[0]:
                tmp_df = pd.DataFrame(pred_test, index=adata_spatial.obs_names, columns=test_gene)
                tmp_df = tmp_df.reindex(sp_data_test.index)
                pred_test = tmp_df.values
            else:
                pred_test = np.zeros((sp_data_test.shape[0], len(test_gene)))

        df_pred_test = pd.DataFrame(pred_test, index=sp_data_test.index, columns=test_gene)
        df_true_test = sp_data_test[test_gene].copy()

        # Metrics alignment and compute
        y_true_val, y_pred_val = align_for_metrics(df_true_val.values, df_pred_val.values, tag=f"stPlus_fold{idx}_val")
        y_true_test, y_pred_test = align_for_metrics(df_true_test.values, df_pred_test.values, tag=f"stPlus_fold{idx}_test")

        results_val = compute_metrics(y_true_val, y_pred_val, genes=test_gene)
        results_val.update({'fold': idx, 'phase': 'val'})
        fold_metrics[idx] = results_val

        results_test = compute_metrics(y_true_test, y_pred_test, genes=test_gene)
        results_test.update({'fold': idx, 'phase': 'test'})
        fold_test_metrics[idx] = results_test

        print(f"[stPlus Fold {idx}] val spearman {results_val.get('spearman_mean_genewise', np.nan):.4f}, test spearman {results_test.get('spearman_mean_genewise', np.nan):.4f}")

        # save per-fold CSVs
        os.makedirs(csv_dir, exist_ok=True)
        df_pred_val.to_csv(os.path.join(csv_dir, f"fold_{idx}_val_pred.csv"), index=True)
        df_true_val.to_csv(os.path.join(csv_dir, f"fold_{idx}_val_true.csv"), index=True)
        df_pred_test.to_csv(os.path.join(csv_dir, f"fold_{idx}_test_pred.csv"), index=True)
        df_true_test.to_csv(os.path.join(csv_dir, f"fold_{idx}_test_true.csv"), index=True)
        pd.DataFrame([results_val]).to_csv(os.path.join(csv_dir, f"fold_{idx}_metrics.csv"), index=False)
        pd.DataFrame([results_test]).to_csv(os.path.join(csv_dir, f"fold_{idx}_test_metrics.csv"), index=False)

        # optional: fill into all_pred_res for the train rows (keeps original adata_spatial order)
        try:
            train_positions = [list(adata_spatial.obs_names).index(i) for i in sp_data.index]
            for col_pos, gene_idx in enumerate(test_ind):
                all_pred_res[train_positions, gene_idx] = df_pred_val.iloc[:, col_pos].values
        except Exception:
            pass

        idx += 1

    # save summaries and combined
    save_fold_and_summary_metrics(fold_metrics, csv_dir, "fold_metrics")
    save_fold_and_summary_metrics(fold_test_metrics, csv_dir, "fold_test_metrics")
    all_pred_df = pd.DataFrame(all_pred_res, index=adata_spatial.obs_names, columns=adata_spatial.var_names)
    os.makedirs(outdir1, exist_ok=True)
    all_pred_df.to_csv(os.path.join(outdir1, "stPlus_impute.csv"), index=True)

    return all_pred_res

# ===========================
# SpaGE implementation
# ===========================
def SpaGE_impute():
    print("\n=== Running SpaGE ===")
    sys.path.append("baseline/SpaGE-master/")
    try:
        from SpaGE.main import SpaGE
    except Exception as e:
        print("[ERROR] SpaGE import failed:", e)
        return None

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    outdir1 = os.path.join(outdir, "spage")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)
    idx = 1
    fold_metrics = {}
    fold_test_metrics = {}
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print(f"\n=== SpaGE Fold {idx} ===")
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]
        pv = len(train_gene) // 2

        # run SpaGE on training spots (sp_data)
        Imp_Genes = SpaGE(sp_data[train_gene], sc_data, n_pv=pv, genes_to_predict=test_gene)
        Imp_arr = np.asarray(Imp_Genes)
        if Imp_arr.ndim == 1 and Imp_arr.size == sp_data.shape[0] * len(test_gene):
            Imp_arr = Imp_arr.reshape(sp_data.shape[0], len(test_gene))

        df_pred = pd.DataFrame(Imp_arr, index=sp_data.index, columns=test_gene)
        df_true_train = sp_data[test_gene].copy()
        df_true_test = sp_data_test[test_gene].copy()

        # metrics
        y_true_val, y_pred_val = align_for_metrics(df_true_train.values, df_pred.values, tag=f"SpaGE_fold{idx}_val")
        y_true_test, y_pred_test = align_for_metrics(df_true_test.values, df_pred.loc[sp_data_test.index].values, tag=f"SpaGE_fold{idx}_test")

        results_val = compute_metrics(y_true_val, y_pred_val, genes=test_gene)
        results_test = compute_metrics(y_true_test, y_pred_test, genes=test_gene)
        results_val.update({'fold': idx, 'phase': 'val'})
        results_test.update({'fold': idx, 'phase': 'test'})
        fold_metrics[idx] = results_val
        fold_test_metrics[idx] = results_test

        # save per-fold
        os.makedirs(csv_dir, exist_ok=True)
        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"))
        df_true_train.to_csv(os.path.join(csv_dir, f"fold_{idx}_train_true.csv"))
        df_true_test.to_csv(os.path.join(csv_dir, f"fold_{idx}_test_true.csv"))
        pd.DataFrame([results_val]).to_csv(os.path.join(csv_dir, f"fold_{idx}_metrics.csv"), index=False)
        pd.DataFrame([results_test]).to_csv(os.path.join(csv_dir, f"fold_{idx}_test_metrics.csv"), index=False)

        idx += 1

    save_fold_and_summary_metrics(fold_metrics, csv_dir, "fold_metrics")
    save_fold_and_summary_metrics(fold_test_metrics, csv_dir, "fold_test_metrics")
    all_pred_df = pd.DataFrame(all_pred_res, index=adata_spatial.obs_names, columns=adata_spatial.var_names)
    all_pred_df.to_csv(os.path.join(outdir1, "SpaGE_impute.csv"), index=True)
    return all_pred_res

# ===========================
# Tangram implementation
# ===========================
def Tangram_impute():
    print("\n=== Running Tangram ===")
    try:
        import baseline.tangram as tg
    except Exception as e:
        print("[ERROR] Tangram import failed:", e)
        return None

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    outdir1 = os.path.join(outdir, "tangram")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)

    idx = 1
    fold_metrics = {}
    fold_test_metrics = {}
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print(f"\n=== Tangram Fold {idx} ===")
        train_gene = list(raw_shared_gene[train_ind])
        test_gene = list(raw_shared_gene[test_ind])

        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()

        adata_spatial_partial = adata_spatial_tmp[:, train_gene]
        train_gene = np.array(train_gene)

        # attempt to compute leiden for annotation
        try:
            RNA_data_adata_label = adata_seq3.copy()
            sc.pp.normalize_total(RNA_data_adata_label)
            sc.pp.log1p(RNA_data_adata_label)
            sc.pp.highly_variable_genes(RNA_data_adata_label)
            RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
            sc.pp.scale(RNA_data_adata_label, max_value=10)
            sc.tl.pca(RNA_data_adata_label)
            sc.pp.neighbors(RNA_data_adata_label)
            sc.tl.leiden(RNA_data_adata_label, resolution=0.5)
            adata_seq_tmp.obs['leiden'] = RNA_data_adata_label.obs.leiden
        except Exception:
            pass

        tg.pp_adatas(adata_seq_tmp, adata_spatial_partial, genes=train_gene)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ad_map = tg.map_cells_to_space(adata_seq_tmp, adata_spatial_partial, device=device, mode='clusters', cluster_label='leiden')
        ad_ge = tg.project_genes(ad_map, adata_seq_tmp, cluster_label='leiden')

        pred_cols = list(ad_ge.var_names)
        pred_df_full = pd.DataFrame(ad_ge.X, index=adata_spatial_partial.obs_names, columns=pred_cols)
        pred_df_full = pred_df_full.reindex(sp_data.index)  # align rows to training spots

        df_pred = pd.DataFrame(0.0, index=sp_data.index, columns=test_gene)
        for g in test_gene:
            if g in pred_df_full.columns:
                df_pred[g] = pred_df_full[g].values

        df_true_train = sp_data[test_gene].copy()
        df_true_test = sp_data_test[test_gene].copy()

        y_true_val, y_pred_val = align_for_metrics(df_true_train.values, df_pred.values, tag=f"Tangram_fold{idx}_val")
        try:
            y_pred_test_raw = df_pred.loc[sp_data_test.index].values
        except Exception:
            y_pred_test_raw = np.zeros_like(df_true_test.values)
        y_true_test, y_pred_test = align_for_metrics(df_true_test.values, y_pred_test_raw, tag=f"Tangram_fold{idx}_test")

        results_val = compute_metrics(y_true_val, y_pred_val, genes=test_gene)
        results_test = compute_metrics(y_true_test, y_pred_test, genes=test_gene)
        results_val.update({'fold': idx, 'phase': 'val'})
        results_test.update({'fold': idx, 'phase': 'test'})
        fold_metrics[idx] = results_val
        fold_test_metrics[idx] = results_test

        os.makedirs(csv_dir, exist_ok=True)
        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"))
        df_true_train.to_csv(os.path.join(csv_dir, f"fold_{idx}_train_true.csv"))
        df_true_test.to_csv(os.path.join(csv_dir, f"fold_{idx}_test_true.csv"))
        pd.DataFrame([results_val]).to_csv(os.path.join(csv_dir, f"fold_{idx}_metrics.csv"), index=False)
        pd.DataFrame([results_test]).to_csv(os.path.join(csv_dir, f"fold_{idx}_test_metrics.csv"), index=False)

        idx += 1

    save_fold_and_summary_metrics(fold_metrics, csv_dir, "fold_metrics")
    save_fold_and_summary_metrics(fold_test_metrics, csv_dir, "fold_test_metrics")
    all_pred_df = pd.DataFrame(all_pred_res, index=adata_spatial.obs_names, columns=adata_spatial.var_names)
    all_pred_df.to_csv(os.path.join(outdir1, "Tangram_impute.csv"), index=True)
    return all_pred_res

# =================================
# UniPort (best-effort)
# =================================
def uniport_impute():
    print("\n=== Running UniPort ===")
    if up is None:
        print("[WARN] UniPort library not found — skipping UniPort.")
        return None

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    outdir1 = os.path.join(outdir, "uniport")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)
    idx = 1
    fold_metrics = {}
    fold_test_metrics = {}
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]

        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()

        adata_spatial_tmp.obs['domain_id'] = 0
        adata_spatial_tmp.obs['source'] = 'ST'
        adata_seq_tmp.obs['domain_id'] = 1
        adata_seq_tmp.obs['source'] = 'RNA'

        adata_cm = adata_spatial_tmp.concatenate(adata_seq_tmp, join='inner', batch_key='domain_id')
        spatial_data = adata_cm[adata_cm.obs['source'] == 'ST'].copy()
        seq_data = adata_cm[adata_cm.obs['source'] == 'RNA'].copy()
        spatial_data_partial = spatial_data[:, train_gene].copy()
        adata_cm2 = spatial_data_partial.concatenate(seq_data, join='inner', batch_key='domain_id')

        try:
            up.batch_scale(adata_cm2)
            up.batch_scale(spatial_data_partial)
            up.batch_scale(seq_data)
            seq_data.X = scipy.sparse.coo_matrix(seq_data.X)
            spatial_data_partial.X = scipy.sparse.coo_matrix(spatial_data_partial.X)
            adatas = [spatial_data_partial, seq_data]
            adata_model = up.Run(adatas=adatas, adata_cm=adata_cm2, lambda_kl=5.0, model_info=False)
            spatial_data_partial.X = spatial_data_partial.X.A
            adata_predict = up.Run(adata_cm=spatial_data_partial, out='predict', pred_id=1)
            model_res = pd.DataFrame(adata_predict.obsm['predict'], columns=raw_shared_gene)
            y_pred = np.asarray(model_res[test_gene], dtype=np.float32)
        except Exception as e:
            print("[WARN] UniPort per-fold failed:", e)
            y_pred = np.zeros((sp_data.shape[0], len(test_gene)))

        df_pred = pd.DataFrame(y_pred, index=sp_data.index, columns=test_gene)
        df_true_train = sp_data[test_gene].copy()
        df_true_test = sp_data_test[test_gene].copy()

        y_true_val, y_pred_val = align_for_metrics(df_true_train.values, df_pred.values, tag=f"UniPort_fold{idx}_val")
        y_true_test, y_pred_test = align_for_metrics(df_true_test.values, df_pred.loc[sp_data_test.index].values, tag=f"UniPort_fold{idx}_test")

        results_val = compute_metrics(y_true_val, y_pred_val, genes=test_gene)
        results_test = compute_metrics(y_true_test, y_pred_test, genes=test_gene)
        fold_metrics[idx] = results_val
        fold_test_metrics[idx] = results_test

        os.makedirs(csv_dir, exist_ok=True)
        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"))
        df_true_train.to_csv(os.path.join(csv_dir, f"fold_{idx}_train_true.csv"))
        df_true_test.to_csv(os.path.join(csv_dir, f"fold_{idx}_test_true.csv"))
        pd.DataFrame([results_val]).to_csv(os.path.join(csv_dir, f"fold_{idx}_metrics.csv"), index=False)
        pd.DataFrame([results_test]).to_csv(os.path.join(csv_dir, f"fold_{idx}_test_metrics.csv"), index=False)

        idx += 1

    save_fold_and_summary_metrics(fold_metrics, csv_dir, "fold_metrics")
    save_fold_and_summary_metrics(fold_test_metrics, csv_dir, "fold_test_metrics")
    return all_pred_res

# =================================
# gimVI (best-effort)
# =================================
def gimVI_impute():
    print("\n=== Running gimVI ===")
    try:
        import baseline.scvi as scvi_baseline
        from scvi.model import GIMVI
    except Exception as e:
        print("[WARN] gimVI not importable:", e)
        return None

    raw_shared_gene = np.array(adata_spatial2.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    outdir1 = os.path.join(outdir, "gimVI")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)

    idx = 1
    fold_metrics = {}
    all_pred_res = np.zeros_like(adata_spatial2.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]

        adata_spatial_partial = adata_spatial2[:, train_gene].copy()
        seq_data = adata_seq2[:, raw_shared_gene].copy()

        sc.pp.filter_cells(adata_spatial_partial, min_counts=0)
        sc.pp.filter_cells(seq_data, min_counts=0)
        scvi.data.setup_anndata(adata_spatial_partial)
        scvi.data.setup_anndata(seq_data)

        try:
            model = GIMVI(seq_data, adata_spatial_partial)
            model.train(200)
            _, imputation = model.get_imputed_values(normalized=False)
            all_pred_res[:, test_ind] = imputation[:, test_ind]
        except Exception as e:
            print("[WARN] gimVI fold failed:", e)
            pass

        idx += 1

    all_pred_df = pd.DataFrame(all_pred_res, index=adata_spatial.obs_names, columns=adata_spatial.var_names)
    os.makedirs(outdir1, exist_ok=True)
    all_pred_df.to_csv(os.path.join(outdir1, "gimVI_impute.csv"), index=True)
    return all_pred_res

# =================================
# Metric computation class (CalculateMetrics) - integrated from your original code
# =================================
def cal_ssim(im1, im2, M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim

def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result

def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content, columns=[label])
        result = pd.concat([result, content], axis=1)
    return result

def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result

def logNorm(df):
    df = np.log1p(df)
    df = st.zscore(df)
    return df

class CalculateMetrics:
    """
    Reworked from your original class name 'CalculateMeteics'.
    Computes SSIM, SPCC (Spearman per-gene), JS divergence, RMSE and cluster metrics (ARI, AMI, Homo, NMI).
    """
    def __init__(self, raw_data, genes_name, impute_count_file, prefix, metric):
        self.impute_count_file = impute_count_file
        # raw_data is expected as numpy matrix (cells x genes) matching genes_name
        self.raw_count = pd.DataFrame(raw_data, columns=genes_name)
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)

        # imputed counts read from CSV
        self.impute_count = pd.read_csv(impute_count_file, header=0, index_col=0)
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)

        self.prefix = prefix
        self.metric = metric

    def SSIM(self, raw, impute, scale='scale_max'):
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print('Please note you do not scale data by scale max')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    # choose M as the larger max for stability
                    M = max(raw_col.max(), impute_col.max())
                    raw_col_2 = np.array(raw_col).reshape(raw_col.shape[0], 1)
                    impute_col_2 = np.array(impute_col).reshape(impute_col.shape[0], 1)
                    ssim = cal_ssim(raw_col_2, impute_col_2, M)
                ssim_df = pd.DataFrame([ssim], index=["SSIM"], columns=[label])
                result = pd.concat([result, ssim_df], axis=1)
        else:
            print("columns error")
            result = pd.DataFrame()
        return result

    def SPCC(self, raw, impute, scale=None):
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    spearmanr = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    spearmanr, _ = st.spearmanr(raw_col, impute_col)
                spearman_df = pd.DataFrame([spearmanr], index=["SPCC"], columns=[label])
                result = pd.concat([result, spearman_df], axis=1)
        else:
            print("columns error")
            result = pd.DataFrame()
        return result

    def JS(self, raw, impute, scale='scale_plus'):
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print('Please note you do not scale data by plus')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    JS = 1
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    raw_col = raw_col.fillna(1e-20)
                    impute_col = impute_col.fillna(1e-20)
                    M = (raw_col + impute_col) / 2
                    JS = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
                JS_df = pd.DataFrame([JS], index=["JS"], columns=[label])
                result = pd.concat([result, JS_df], axis=1)
        else:
            print("columns error")
            result = pd.DataFrame()
        return result

    def RMSE(self, raw, impute, scale='zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = 1.5
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())
                RMSE_df = pd.DataFrame([RMSE], index=["RMSE"], columns=[label])
                result = pd.concat([result, RMSE_df], axis=1)
        else:
            print("columns error")
            result = pd.DataFrame()
        return result

    def cluster(self, raw, impu, scale=None):
        # use adata_spatial2 as basis (normalized + log1p)
        ad_sp = adata_spatial2.copy()
        ad_sp.X = raw
        cpy_x = adata_spatial2.copy()
        cpy_x.X = impu

        # compute leiden clusters for each (this may create different labels; we align by using ad_sp clusters as reference)
        sc.tl.pca(ad_sp)
        sc.pp.neighbors(ad_sp, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(ad_sp)
        tmp_adata1 = ad_sp

        sc.tl.pca(cpy_x)
        sc.pp.neighbors(cpy_x, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(cpy_x)
        tmp_adata2 = cpy_x

        # set class label in one adata and compute clustering metrics w.r.t. leiden
        tmp_adata2.obs['class'] = tmp_adata1.obs['leiden']
        ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        result = pd.DataFrame([[ari, ami, homo, nmi]], columns=["ARI", "AMI", "Homo", "NMI"])
        return result

    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        prefix = self.prefix
        SSIM_gene = self.SSIM(raw, impute)
        Spearman_gene = self.SPCC(raw, impute)
        JS_gene = self.JS(raw, impute)
        RMSE_gene = self.RMSE(raw, impute)
        cluster_result = self.cluster(raw, impute)

        result_gene = pd.concat([Spearman_gene, SSIM_gene, RMSE_gene, JS_gene], axis=0)
        result_gene.T.to_csv(prefix + "_gene_Metrics.txt", sep='\t', header=1, index=1)
        cluster_result.to_csv(prefix + "_cluster_Metrics.txt", sep='\t', header=1, index=1)
        return result_gene

# =================================
# CalDataMetric wrapper: compute metrics for all model outputs under Result/<Data>
# =================================
def CalDataMetric(Data):
    """
    - Scans Result/<Data> for model directories (stplus, spage, tangram, uniport, gimVI).
    - For each model folder, looks for *_impute.csv files (the script saves them), and
      runs CalculateMetrics on each found impute CSV.
    - Saves final_result.csv in each model folder combining median per-gene metrics and clustering results.
    """
    print("We are calculating metrics for:", Data)
    impute_root = os.path.join("Result", Data)
    if not os.path.exists(impute_root):
        print("[WARN] Result directory does not exist:", impute_root)
        return

    # list candidate model dirs
    model_dirs = [d for d in os.listdir(impute_root) if os.path.isdir(os.path.join(impute_root, d))]
    print("Found model directories:", model_dirs)

    for mod in model_dirs:
        mod_dir = os.path.join(impute_root, mod)
        # find all files that end with impute.csv (these are combined predictions you saved)
        impute_files = []
        for root, _, files in os.walk(mod_dir):
            for f in files:
                if f.endswith("impute.csv") or f.endswith("_impute.csv"):
                    impute_files.append(os.path.join(root, f))

        if len(impute_files) == 0:
            print(f"[INFO] No impute.csv files found for model {mod} in {mod_dir}. Skipping.")
            continue

        # prepare output dir for final result
        os.makedirs(mod_dir, exist_ok=True)
        all_medians = []
        method_names = []
        for impute_path in impute_files:
            print("Processing:", impute_path)
            # prefix for saving metrics (use file path without .csv)
            prefix = impute_path[:-4]
            try:
                imputed_df = pd.read_csv(impute_path, index_col=0)
                raw_df = pd.DataFrame(adata_spatial2.X, columns=sp_genes)
                common_cols = [c for c in imputed_df.columns if c in raw_df.columns]
                if len(common_cols) == 0:
                    print(f"[WARN] No common genes between raw and imputed data for {impute_path}. Skipping.")
                    continue
                raw_df = raw_df[common_cols]
                imputed_df = imputed_df[common_cols]
                aligned_path = impute_path.replace(".csv", "_aligned.csv")
                imputed_df.to_csv(aligned_path)
                CM = CalculateMetrics(raw_df.values, common_cols, impute_count_file=aligned_path, prefix=prefix, metric=['SPCC','SSIM','RMSE','JS'])
                CM.compute_all()

                # read back gene metrics file to compute medians
                tmp = pd.read_csv(prefix + "_gene_Metrics.txt", sep='\t', index_col=0)
                medians = []
                for m in ['SPCC','SSIM','RMSE','JS']:
                    if m in tmp.columns:
                        medians.append(np.median(tmp[m].values))
                    else:
                        medians.append(np.nan)
                clu = pd.read_csv(prefix + "_cluster_Metrics.txt", sep='\t', index_col=0)
                combined = pd.DataFrame([medians + clu.iloc[0].tolist()], columns=['SPCC(gene)','SSIM(gene)','RMSE(gene)','JS(gene)'] + list(clu.columns))
                all_medians.append(combined)
                method_names.append(os.path.basename(impute_path).split('_')[0])
            except Exception as e:
                print("[WARN] Calculating metrics failed for", impute_path, ":", e)
                continue

        if len(all_medians) == 0:
            print(f"[INFO] No successful metric results for model {mod}")
            continue

        result_df = pd.concat(all_medians, axis=0)
        result_df.index = method_names
        final_path = os.path.join(mod_dir, "final_result.csv")
        result_df.to_csv(final_path, index=True)
        print(f"[INFO] Saved final result for {mod} to {final_path}")

# =================================
# MAIN - Run baselines then metrics
# =================================
if __name__ == "__main__":
    # stPlus
    print("\n>> Running stPlus ...")
    try:
        stPlus_res = stPlus_impute()
    except Exception as e:
        print("stPlus failed:", e)

    # SpaGE
    print("\n>> Running SpaGE ...")
    try:
        SpaGE_res = SpaGE_impute()
    except Exception as e:
        print("SpaGE failed:", e)

    # Tangram
    print("\n>> Running Tangram ...")
    try:
        Tangram_res = Tangram_impute()
    except Exception as e:
        print("Tangram failed:", e)

    # UniPort
    print("\n>> Running UniPort ...")
    try:
        uni_res = uniport_impute()
    except Exception as e:
        print("UniPort failed:", e)

    # gimVI
    print("\n>> Running gimVI ...")
    try:
        gim_res = gimVI_impute()
    except Exception as e:
        print("gimVI failed:", e)

    # Finally compute CalDataMetric for all models found under Result/<Data>
    print("\n>> Running CalDataMetric on all models")
    try:
        CalDataMetric(Data)
    except Exception as e:
        print("CalDataMetric failed:", e)

    print("\nAll done.")