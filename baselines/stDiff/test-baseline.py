import numpy as np
import pandas as pd
import random
import sys
import os
import scipy.stats as st
import copy
from sklearn.model_selection import KFold
import pandas as pd
import scanpy as sc
import warnings
from os.path import join
from sklearn.model_selection import train_test_split
import torch
from utils.compute_metrics import compute_metrics, spearmanrr
warnings.filterwarnings('ignore')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
from process.result_analysis import clustering_metrics

from process.result_analysis import *
# from baseline.scvi.model import GIMVI
from baseline.stPlus import *
from process.data import *


import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default="dataset5_seq_915.h5ad")
parser.add_argument("--sp_data", type=str, default='dataset5_spatial_915.h5ad')
# parser.add_argument("--sp_data_test", type=str, default='dataset5_spatial_915.h5ad')
parser.add_argument("--document", type=str, default='dataset5')
parser.add_argument("--filename", type=str, default='Dataset')
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_args()
# ******** preprocess ********


def set_all_seeds(seed: int = 42):
    """
    Fixes all random seeds for reproducibility across:
    - Python's random module
    - NumPy
    - PyTorch (CPU and GPU)
    - OS-level hash seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: use deterministic algorithms globally (PyTorch ≥ 1.8)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    print(f"[Seed fixed] All seeds set to {seed}")

# set_all_seeds(42)

n_splits = 5 
print("adata_spatial: ", args.sp_data)
print("adata_seq: ", args.sc_data)

adata_spatial = sc.read_h5ad(args.sp_data)
adata_seq = sc.read_h5ad(args.sc_data)

# adata_spatial = adata_spatial[:,:].copy()
# adata_seq = adata_seq[:,:].copy()

print(f"sp shape {adata_spatial.shape}")
print(f"sc shape {adata_seq.shape}")

# =========== Amit Preprocessing ==================

sc.pp.filter_cells(adata_seq, min_counts=1)
sc.pp.filter_genes(adata_seq, min_counts=1)
sc.pp.filter_cells(adata_spatial, min_counts=1)
sc.pp.filter_genes(adata_spatial, min_counts=1)

common_genes = np.intersect1d(adata_spatial.var_names, adata_seq.var_names)

adata_spatial = adata_spatial[:, common_genes].copy()
adata_seq = adata_seq[:, common_genes].copy()

#=============================================================
print(f"The number of common genes:: {len(common_genes)}")

adata_seq2 = adata_seq.copy()
# tangram
adata_seq3 =  adata_seq2.copy()
sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
data_seq_array = adata_seq2.X

adata_spatial2 = adata_spatial.copy()
sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
data_spatial_array = adata_spatial2.X

sp_genes = np.array(adata_spatial.var_names)
sp_data = pd.DataFrame(data=data_spatial_array, columns=sp_genes)
sc_data = pd.DataFrame(data=data_seq_array, columns=sp_genes)

print(f"sp_data: {sp_data.shape}, sc_data: {sc_data.shape}")

# ==============================================================
# 80/20 spatial cell split
# ==============================================================

# train_spatial_df, test_spatial_df = train_test_split(
#     sp_data, test_size=0.2, random_state=42, shuffle=True
# )
# print(f"Spatial split -> Train: {train_spatial_df.shape}, Test: {test_spatial_df.shape}")


sp_idx_train, sp_idx_test = train_test_split(
    np.arange(adata_spatial2.n_obs),
    test_size=0.2,
    random_state=42,
    shuffle=True
)


np.save(f"/home/maninder/AmitCode/stDiff/process/{args.filename}_train_indices.npy", sp_idx_train)
np.save(f"/home/maninder/AmitCode/stDiff/process/{args.filename}_test_indices.npy", sp_idx_test)

print(f"sp_idx_test: {sp_idx_test}")

adata_seq_train = adata_spatial2[sp_idx_train].copy()
adata_seq_test = adata_spatial2[sp_idx_test].copy()

print(f"scRNA split -> Train: {adata_seq_train.shape}, Test: {adata_seq_test.shape}")

sp_data_train = pd.DataFrame(data=adata_seq_train.X, columns=sp_genes)
sp_data_test = pd.DataFrame(data=adata_seq_test.X, columns=sp_genes)



# sp_data = train_spatial_df.copy()
# sp_data_test = test_spatial_df.copy()

# outdir = os.path.join("Result", args.filename, "UniPort")
# csv_dir = os.path.join(outdir, "fold_results")
# os.makedirs(csv_dir, exist_ok=True)

# ****baseline****

def SpaGE_impute():
    '''
    需要预处理
    Returns
    -------

    '''
    print ('We run SpaGE for this data\n')
    sys.path.append("baseline/SpaGE-master/")
    from SpaGE.main import SpaGE
    global sc_data, sp_data, adata_seq, adata_spatial, adata_seq_test


    outdir1 = os.path.join("Result", args.document, "spage")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)


    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_seq_test.X)
    fold_metrics = {}
    gene_seq = []
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]
        pv = len(train_gene) // 2
        sp_data_partial = sp_data_test[train_gene]
        gene_seq.append(test_gene.tolist())

        print(f"gene_seq:: {gene_seq}")

        Imp_Genes = SpaGE(sp_data_partial, sc_data, n_pv=50,
                          genes_to_predict=test_gene)

        all_pred_res[:, test_ind] = Imp_Genes
        
        #============ Amit Added Code =============
        y_true = sp_data_test[test_gene].values
        y_pred = np.asarray(Imp_Genes)

        results_val = compute_metrics(y_true, y_pred, genes=test_gene)
        results_val.update({
            'fold': idx,
            'phase': 'val',
            'loss': float(results_val.get('l2_errors_mean', 0.0))
        })

        fold_metrics[idx] = results_val

        print(results_val)
        print(f"[Fold {idx}] SpaGE val Spearman: {results_val['spearman_mean_genewise']:.4f}, "
                        f"Pearson: {results_val['pearson_mean']:.4f}")
        #============ Amit Added Code =============

        # Save per-fold predictions and ground truth
        df_pred = pd.DataFrame(Imp_Genes.to_numpy(), columns=test_gene)
        df_true = pd.DataFrame(y_true, columns=test_gene)
        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"), index=False)
        df_true.to_csv(os.path.join(csv_dir, f"fold_{idx}_true.csv"), index=False)

        idx += 1

    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(os.path.join(csv_dir, "fold_metrics.csv"), index=False)
    print(f"Saved all fold metrics to {csv_dir}/fold_metrics.csv")

    # Compute mean and std
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    # Combine into one summary table
    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(os.path.join(csv_dir, "metrics_summary.csv"), index=False)
    print(f"Saved mean and std metrics to {csv_dir}/metrics_summary.csv")

    print(f"all_pred_res :::: {all_pred_res.shape}")

    return all_pred_res, gene_seq



######### Tangram Code #############################################


def Tangram_impute(annotate=None, modes='clusters', density='rna_count_based'):
    '''
    Returns
    -------

    '''


    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    import baseline.tangram as tg
    print('We run Tangram for this data\n')
    global adata_seq3, adata_spatial, locations, adata_seq_test, sp_idx_test, sp_data_test
    from sklearn.model_selection import KFold

    fold_metrics = {}
    outdir1 = os.path.join("Result", args.document, "tangram")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)
    
    raw_shared_gene = adata_spatial.var_names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1

    all_pred_res = np.zeros_like(adata_seq_test.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = list(raw_shared_gene[train_ind])
        test_gene = list(raw_shared_gene[test_ind])

        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()

        adata_spatial_partial = adata_spatial_tmp[:, train_gene]
        train_gene = np.array(train_gene)
        if annotate == None:
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
        else:
            global CellTypeAnnotate
            adata_seq_tmp.obs['leiden'] = CellTypeAnnotate
        tg.pp_adatas(adata_seq_tmp, adata_spatial_partial, genes=train_gene) 

        device = torch.device('cuda')
        if modes == 'clusters':
            ad_map = tg.map_cells_to_space(adata_seq_tmp, adata_spatial_partial, device=device, mode=modes,
                                           cluster_label='leiden', density_prior=density)
            ad_ge = tg.project_genes(ad_map, adata_seq_tmp, cluster_label='leiden')
        else:
            ad_map = tg.map_cells_to_space(adata_seq_tmp, adata_spatial_partial, device=device)
            ad_ge = tg.project_genes(ad_map, adata_seq_tmp)
        test_list = list(set(ad_ge.var_names) & set(test_gene))
        test_list = np.array(test_list)

        #Changes===================
        y_pred = np.asarray(ad_ge.X[np.ix_(sp_idx_test, test_ind)], dtype=np.float32)
        y_true = np.asarray(adata_seq_test[:, test_gene].X, dtype=np.float32)
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        #==================================

        all_pred_res[:, test_ind] = ad_ge.X[np.ix_(sp_idx_test, test_ind)]

        results_val = compute_metrics(y_true, y_pred, genes=test_gene)
        results_val = {
            'fold': idx,
            'phase': 'val',
            'loss': float(results_val.get('l2_errors_mean', 0.0)),
            **results_val
        }
        fold_metrics[idx] = results_val

        print(results_val)

        print(f"[Fold {idx}] Spearman: {results_val['spearman_mean_genewise']:.4f}, "
              f"Pearson: {results_val['pearson_mean']:.4f}, "
              f"L2 loss: {results_val['l2_errors_mean']:.4f}")

        # Save per-fold predictions and ground truth
        df_pred = pd.DataFrame(y_pred, columns=test_gene)
        df_true = sp_data_test
        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"), index=False)
        df_true.to_csv(os.path.join(csv_dir, f"fold_{idx}_true.csv"), index=False)
        idx += 1

    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(os.path.join(csv_dir, "fold_metrics.csv"), index=False)
    print(f"Saved all fold metrics to {csv_dir}/fold_metrics.csv")

    # Compute mean and std
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    # Combine into one summary table
    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(os.path.join(csv_dir, "metrics_summary.csv"), index=False)
    print(f"Saved mean and std metrics to {csv_dir}/metrics_summary.csv")




    return all_pred_res



def stPlus_impute():
    '''
    输入预处理后的数据，需要标准预处理
    Returns
    -------

    '''
    global sc_data, sp_data, outdir, adata_spatial, sp_idx_test, sp_data_test


    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    
    outdir1 = os.path.join("Result", args.document, "stplus")
    csv_dir = os.path.join(outdir1, "fold_results")

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"outdir1: {outdir1}")
    os.makedirs(csv_dir, exist_ok=True)
    
    fold_metrics={}
    idx = 1
    # all_pred_res = np.zeros_like(adata_spatial.X)
    all_pred_res = np.zeros_like(adata_seq_test.X)
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])

        save_path_prefix = join(outdir, 'process_file/stPlus-demo')
        # if not os.path.exists(join(outdir, "process_file")):
            # os.mkdir(join(outdir, "process_file"))
        os.makedirs(f"{outdir}/process_file/stPlus-demo", exist_ok=True)

        stPlus_res = stPlus(sp_data[train_gene], sc_data, test_gene, save_path_prefix)
        print(f"stPlus_res type ++++++++++++++++++++++++++ : {type(stPlus_res)}")
        # print(f"shape stPlus_res {stPlus_res}")
        # all_pred_res[:, test_ind] = stPlus_res.iloc[sp_idx_test, :].to_numpy()
        stPlus_df = stPlus_res[1]
        print("stPlus_df: ", stPlus_df.shape) 
        print("stPlus_df type: ", type(stPlus_df)) 
        print("all_pred_res shape: ", all_pred_res.shape) 
        # print("all_pred_res type: ", type((all_pred_res)) 

        # print("stPlus_df updated ::  ", stPlus_df.iloc[sp_idx_test, test_ind].shape)
        # print(f"sp_idx_test {sp_idx_test}")
        # print(f"test_ind {test_ind}")
        rows = np.array(sp_idx_test).flatten()
        cols = np.array(test_ind).flatten()

        # print(f"row len :: {len(rows)}")
        # print(f"cols len :: {len(cols)}")
        my_test = stPlus_df.iloc[rows, :]
        # print(f"shape of my_test: ", my_test.shape)
        # my_test = stPlus_df.iloc[rows, cols].to_numpy()

        # print(f"sp_data+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ : {type(sp_data)}")
        df_true = sp_data.iloc[rows, :] 
        # print(f"sp_data : {type(sp_data)}")

        print(f"my_test {my_test.shape}")

        all_pred_res[:, test_ind] = my_test.to_numpy()

        results_val = compute_metrics(df_true[test_gene].to_numpy(), my_test.to_numpy(), genes=test_gene)
        results_val = {
            'fold': idx,
            'phase': 'val',
            'loss': float(results_val.get('l2_errors_mean', 0.0)),
            **results_val
        }
        fold_metrics[idx] = results_val

        print(results_val)
        
        df_pred = my_test
        # df_true = pd.DataFrame(adata_spatial[:, test_ind].X, columns=test_gene)
        df_true = sp_data[test_gene]



        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"), index=False)
        df_true.to_csv(os.path.join(csv_dir, f"fold_{idx}_true.csv"), index=False)
        idx += 1
    
    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(os.path.join(csv_dir, "fold_metrics.csv"), index=False)
    print(f"Saved all fold metrics to {csv_dir}/fold_metrics.csv")

    # Compute mean and std
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    # Combine into one summary table
    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(os.path.join(csv_dir, "metrics_summary.csv"), index=False)
    print(f"Saved mean and std metrics to {csv_dir}/metrics_summary.csv")

    return all_pred_res


def gimVI_impute():
    '''
    本来处理的就是原始数据，所以传入预处理或者原始数据都可

    如果是标准与处理 需要传入adata_seq2 spatial2
    Returns
    -------

    '''
    print ('We run gimVI for this data\n')
    import baseline.scvi as scvi
    import scanpy as sc
    # from scvi.model import GIMVI
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    global adata_seq2, adata_spatial2

    from sklearn.model_selection import KFold
    raw_shared_gene = adata_spatial2.var_names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)  # shuffle = false 不设置state，就是按顺序划分
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(data_spatial_array)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])

        Genes = list(adata_spatial2.var_names)
        rand_gene_idx = test_ind
        n_genes = len(Genes)
        rand_train_gene_idx = train_ind
        rand_train_genes = np.array(Genes)[rand_train_gene_idx] # 不就是train_genes吗
        rand_genes = np.array(Genes)[rand_gene_idx] # test_gene
        adata_spatial_partial = adata_spatial2[:, rand_train_genes]
        sc.pp.filter_cells(adata_spatial_partial, min_counts=0)
        seq_data = copy.deepcopy(adata_seq2)
        seq_data = seq_data[:, Genes]
        sc.pp.filter_cells(seq_data, min_counts=0)
        scvi.data.setup_anndata(adata_spatial_partial)
        scvi.data.setup_anndata(seq_data)
        model = GIMVI(seq_data, adata_spatial_partial)
        model.train(200)
        _, imputation = model.get_imputed_values(normalized=False)
        all_pred_res[:, test_ind] = imputation[:, rand_gene_idx]
        idx += 1

    return all_pred_res





# def uniport_impute():
#     global sc_data, sp_data, adata_seq, adata_spatial

#     outdir = os.path.join("Result", args.filename, "uniport")
#     csv_dir = os.path.join(outdir, "fold_results")
#     os.makedirs(csv_dir, exist_ok=True)


#     raw_shared_gene = np.array(adata_spatial.var_names)
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand) # 0
#     kf.get_n_splits(raw_shared_gene)
#     torch.manual_seed(10)
#     idx = 1
#     all_pred_res = np.zeros_like(adata_spatial.X)

#     for train_ind, test_ind in kf.split(raw_shared_gene):
#         print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
#             idx, len(train_ind), len(test_ind)))
#         train_gene = raw_shared_gene[train_ind]
#         test_gene = raw_shared_gene[test_ind]
        
#         adata_seq_tmp = adata_seq2.copy()
#         adata_spatial_tmp = adata_spatial2.copy()
        
#         adata_spatial_tmp.obs['domain_id'] = 0
#         adata_spatial_tmp.obs['domain_id'] = adata_spatial_tmp.obs['domain_id'].astype('category')
#         adata_spatial_tmp.obs['source'] = 'ST'

#         adata_seq_tmp.obs['domain_id'] = 1
#         adata_seq_tmp.obs['domain_id'] = adata_seq_tmp.obs['domain_id'].astype('category')
#         adata_seq_tmp.obs['source'] = 'RNA'
        
#         adata_cm = adata_spatial_tmp.concatenate(adata_seq_tmp, join='inner', batch_key='domain_id')
#         print(adata_cm.obs)
#         spatial_data = adata_cm[adata_cm.obs['source']=='ST'].copy()
#         seq_data = adata_cm[adata_cm.obs['source']=='RNA'].copy()
        
#         spatial_data_partial = spatial_data[:,train_gene].copy()
#         adata_cm = spatial_data_partial.concatenate(seq_data, join='inner', batch_key='domain_id')
#         print(adata_cm.X.shape)
#         # return
#         up.batch_scale(adata_cm)
#         up.batch_scale(spatial_data_partial)
#         up.batch_scale(seq_data)
        
#         seq_data.X = scipy.sparse.coo_matrix(seq_data.X)
#         spatial_data_partial.X = scipy.sparse.coo_matrix(spatial_data_partial.X)

#         adatas = [spatial_data_partial, seq_data]

#         adata = up.Run(adatas=adatas, adata_cm=adata_cm, lambda_kl=5.0, model_info=False)

#         spatial_data_partial.X = spatial_data_partial.X.A

#         adata_predict = up.Run(adata_cm=spatial_data_partial, out='predict', pred_id=1)
#         model_res = pd.DataFrame(adata_predict.obsm['predict'], columns=raw_shared_gene)

#         all_pred_res[:, test_ind] = model_res[test_gene]
#         idx += 1

#     return all_pred_res



def uniport_impute():
    """
    UniPort imputation baseline, split by gene count (like Tangram).
    Saves per-fold predictions, ground truth, and summary metrics.
    """
    import scipy
    global sc_data, sp_data, adata_seq, adata_spatial, adata_seq2, adata_spatial2

    print("We run UniPort for this data\n")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.set_default_tensor_type('torch.FloatTensor')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.set_default_device("cuda")


    outdir1 = os.path.join("Result", args.document, "uniport")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)

    # Get all shared genes between ST and scRNA-seq
    raw_shared_gene = np.array(adata_spatial.var_names)
    print(f"Total shared genes for KFold split: {len(raw_shared_gene)}")

    # Split based on gene count
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1

    # Store all results
    all_pred_res = np.zeros_like(adata_spatial.X)
    fold_metrics = {}
    # --- Loop over folds ---
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print(f"\n===== Fold {idx} =====")
        print(f"Number of train genes: {len(train_ind)}, Number of test genes: {len(test_ind)}")

        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]

        # --- Prepare adatas ---
        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()

        adata_spatial_tmp.obs['domain_id'] = 0
        adata_spatial_tmp.obs['domain_id'] = adata_spatial_tmp.obs['domain_id'].astype('category')
        adata_spatial_tmp.obs['source'] = 'ST'

        adata_seq_tmp.obs['domain_id'] = 1
        adata_seq_tmp.obs['domain_id'] = adata_seq_tmp.obs['domain_id'].astype('category')
        adata_seq_tmp.obs['source'] = 'RNA'

        # Combine both modalities
        adata_cm = adata_spatial_tmp.concatenate(adata_seq_tmp, join='inner', batch_key='domain_id')
        print(f"[Fold {idx}] Combined AnnData shape: {adata_cm.shape}")

        # Split spatial and RNA subsets
        spatial_data = adata_cm[adata_cm.obs['source'] == 'ST'].copy()
        seq_data = adata_cm[adata_cm.obs['source'] == 'RNA'].copy()

        # Train on a subset of genes
        spatial_data_partial = spatial_data[:, train_gene].copy()

        # Combine for UniPort input
        adata_cm = spatial_data_partial.concatenate(seq_data, join='inner', batch_key='domain_id')
        print(f"[Fold {idx}] Combined shape after train gene selection: {adata_cm.shape}")

        # --- Preprocessing ---
        up.batch_scale(adata_cm)
        up.batch_scale(spatial_data_partial)
        up.batch_scale(seq_data)

        seq_data.X = scipy.sparse.coo_matrix(seq_data.X)
        spatial_data_partial.X = scipy.sparse.coo_matrix(spatial_data_partial.X)

        adatas = [spatial_data_partial, seq_data]

        # --- Train UniPort ---
        adata = up.Run(adatas=adatas, adata_cm=adata_cm, lambda_kl=5.0, model_info=False)

        spatial_data_partial.X = spatial_data_partial.X.A

        # --- Predict unseen genes ---
        adata_predict = up.Run(adata_cm=spatial_data_partial, out='predict', pred_id=1)
        model_res = pd.DataFrame(adata_predict.obsm['predict'], columns=raw_shared_gene)

        # --- Save predictions ---
        y_pred = np.asarray(model_res[test_gene], dtype=np.float32)
        y_true = np.asarray(adata_spatial_tmp[:, test_gene].X, dtype=np.float32)
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)

        all_pred_res[:, test_ind] = y_pred

        # --- Compute metrics ---
        results_val = compute_metrics(y_true, y_pred, genes=test_gene)
        results_val.update({
            'fold': idx,
            'phase': 'val',
            'loss': float(results_val.get('l2_errors_mean', 0.0))
        })
        fold_metrics[idx] = results_val

        print(results_val)
        print(f"[Fold {idx}] Spearman: {results_val['spearman_mean_genewise']:.4f}, "
              f"Pearson: {results_val['pearson_mean']:.4f}, "
              f"L2 loss: {results_val['l2_errors_mean']:.4f}")

        # --- Save per-fold results ---
        df_pred = pd.DataFrame(y_pred, columns=test_gene)
        df_true = pd.DataFrame(y_true, columns=test_gene)
        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"), index=False)
        df_true.to_csv(os.path.join(csv_dir, f"fold_{idx}_true.csv"), index=False)

        idx += 1

    # --- Save all fold metrics ---
    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(os.path.join(csv_dir, "fold_metrics.csv"), index=False)
    print(f"Saved all fold metrics to {csv_dir}/fold_metrics.csv")

    # --- Compute mean and std ---
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    # Combine and save
    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(os.path.join(csv_dir, "metrics_summary.csv"), index=False)
    print(f"Saved mean and std metrics to {csv_dir}/metrics_summary.csv")

    print("\n UniPort imputation complete.")
    return all_pred_res



Data = args.document
outdir = 'Result/' + Data + '/'
if not os.path.exists(outdir):
    # os.mkdir(outdir)
    os.makedirs(outdir, exist_ok=True)



SpaGE_result, gene_seq_test = SpaGE_impute() 
SpaGE_result_pd = pd.DataFrame(SpaGE_result, columns=sp_genes)
os.makedirs(f"{outdir}/spage",  exist_ok=True)
SpaGE_result_pd.to_csv(outdir +  'spage/SpaGE_impute.csv',header = 1, index = 1)
print(f"{outdir}/SpaGE_impute.csv")

Tangram_result = Tangram_impute() 
Tangram_result_pd = pd.DataFrame(Tangram_result, columns=sp_genes)
os.makedirs(f"{outdir}/tangram",  exist_ok=True)
Tangram_result_pd.to_csv(outdir +  'tangram/Tangram_impute.csv',header = 1, index = 1)

# gimVI_result = gimVI_impute() 
# gimVI_result_pd = pd.DataFrame(gimVI_result, columns=sp_genes)
# gimVI_result_pd.to_csv(outdir +  '/gimVI_impute.csv',header = 1, index = 1)


stPlus_result = stPlus_impute() 
stPlus_result_pd = pd.DataFrame(stPlus_result, columns=sp_genes)
stPlus_result_pd.to_csv(outdir +  '/stPlus_impute.csv',header = 1, index = 1)

# uniport_result = uniport_impute()
# uniport_result_pd = pd.DataFrame(uniport_result, columns=sp_genes)
# os.makedirs(f"{outdir}/uniport",  exist_ok=True)
# uniport_result_pd.to_csv(outdir +  'uniport/uniport_impute.csv',header = 1, index = 1)

#******** metrics ********

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


class CalculateMeteics:
    def __init__(self, raw_data, gene_seq_test, genes_name,impute_count_file, prefix, metric):
        self.impute_count_file = impute_count_file
        self.gene_seq_test = gene_seq_test
        self.raw_count = pd.DataFrame(raw_data, columns=genes_name)

        print(f"raw_count :::::  {self.raw_count.shape}")
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)

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

            print(f"raw = {raw.shape}")
            print(f"impute = {impute.shape}")
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
                    M = [raw_col.max(), impute_col.max()][raw_col.max() > impute_col.max()]
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)
                    ssim = cal_ssim(raw_col_2, impute_col_2, M)

                ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
                result = pd.concat([result, ssim_df], axis=1)
        else:
            print("columns error")
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
                spearman_df = pd.DataFrame(spearmanr, index=["SPCC"], columns=[label])
                result = pd.concat([result, spearman_df], axis=1)
        else:
            print("columns error")
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
                JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
                result = pd.concat([result, JS_df], axis=1)
        else:
            print("columns error")
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

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"], columns=[label])
                result = pd.concat([result, RMSE_df], axis=1)
        else:
            print("columns error")
        return result


    def cluster(self, raw, impu,scale=None):

        ad_sp = adata_spatial2[sc_idx_test].copy()
        ad_sp.X = raw

        cpy_x = adata_spatial2[sc_idx_test].copy()
        cpy_x.X = impu

        sc.tl.pca(ad_sp)
        sc.pp.neighbors(ad_sp, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(ad_sp)
        tmp_adata1 = ad_sp

        sc.tl.pca(cpy_x)
        sc.pp.neighbors(cpy_x, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(cpy_x)
        tmp_adata2 = cpy_x

        tmp_adata2.obs['class'] = tmp_adata1.obs['leiden']
        ari = clustering_metrics(tmp_adata2, gene_seq_test, 'class', 'leiden', "ARI")
        ami = clustering_metrics(tmp_adata2, gene_seq_test, 'class', 'leiden', "AMI")
        homo = clustering_metrics(tmp_adata2, gene_seq_test, 'class', 'leiden', "Homo")
        nmi = clustering_metrics(tmp_adata2, gene_seq_test,'class', 'leiden', "NMI")
        
        
        # tmp_adata2 = get_N_clusters(cpy_x, 23, 'leiden')
        
        # tmp_adata2.obs['class'] = ad_sp.obs['subclass_label']
        # ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        # ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        # homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        # nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        
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


import seaborn as sns
import os
PATH = 'Result/'
DirFiles = os.listdir(PATH)
# os.makedirs(output_dir, exist_ok=True)


from pathlib import Path
def CalDataMetric(Data, gene_seq_test):
    print ('We are calculating the : ' + Data + '\n')
    impute_count_dir = PATH + Data

    print(f"impute_count_dir++++++++++++++++++++++++++++++++++++++++++++++++++{impute_count_dir}")

    model_list = ['spage','stplus','tangram']
    for index, mod in enumerate(model_list):

        metrics = ['SPCC(gene)','SSIM(gene)','RMSE(gene)','JS(gene)']
        metric = ['SPCC','SSIM','RMSE','JS']
        if index != 0:
            impute_count_dir = os.path.dirname(impute_count_dir)

        if not os.path.exists(f"{impute_count_dir}/{mod}"):
            continue
        impute_count = os.listdir(f"{impute_count_dir}/{mod}")

        print(f"impute_count::: {impute_count}")
        impute_count_dir = f"{impute_count_dir}/{mod}"
        # impute_count = os.listdir(f"{impute_count_dir}/{mod}")
        impute_count_dir = f"{impute_count_dir}"



        # print(f"{os.listdir(impute_count_dir)}")
        print(f"+++++++++++++++++++++++++++{impute_count}++++++++")
        impute_count = [x for x in impute_count if x [-10:] == 'impute.csv']
        print(f"impute_count: {impute_count}")
        methods = []
        if len(impute_count)!=0:
            medians = pd.DataFrame()
            for impute_count_file in impute_count:
                print(impute_count_file)
                if 'result_Tangram.csv' in impute_count_file:
                    os.system('mv ' + impute_count_dir + '/result_Tangram.csv ' + impute_count_dir + '/Tangram_impute.csv')
                prefix = impute_count_file.split('_')[0]
                methods.append(prefix)
                prefix = impute_count_dir + '/' + prefix
                impute_count_file = impute_count_dir + '/' + impute_count_file
                # if not os.path.isfile(prefix + '_Metrics.txt'):
                print (impute_count_file)
                CM = CalculateMeteics(adata_seq_test.X, gene_seq_test, sp_genes, impute_count_file = impute_count_file, prefix = prefix, metric = metric)
                CM.compute_all()

                median = []
                for j in ['_gene']:
                # j = '_gene'
                #     median = []
                    tmp = pd.read_csv(prefix + j + '_Metrics.txt', sep='\t', index_col=0)
                    for m in metric:
                        median.append(np.median(tmp[m]))

                    print(f"median ::: {median}")
                    print(f"metrics ::: {metrics}")
                median = pd.DataFrame([median], columns=metrics)
                # 聚类指标
                clu = pd.read_csv(prefix + '_cluster' + '_Metrics.txt', sep='\t', index_col=0)

                print("")
                median = pd.concat([median, clu], axis=1)
                medians = pd.concat([medians, median], axis=0)

            metrics += ["ARI", "AMI", "Homo", "NMI"]
            medians.columns = metrics
            medians.index = methods
            medians.to_csv(outdir +  f'{mod}/final_result.csv',header = 1, index = 1)
            del metrics

# CalDataMetric(Data, gene_seq_test)
