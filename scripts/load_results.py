from pathlib import Path
import os
import math
import pandas as pd
import numpy as np
import json
import glob
from IPython import embed
import sys
from collections import defaultdict
import collections

from matplotlib import pyplot as plt
from util import round_significant


def highlight_best_with_std(df, larger_is_better_dict, top=2, small_is_better=False):
    """ actually operates on dataframes
        here, takes df with mean (to determine best), and one with std as to reformat
    """
    formats = [[r' \underline{\textbf{', '}}'],
               [r' \textbf{', '}']]

    for col in df.columns:
        # as pm formatting occured before, extract means:
        means = df[col].str.split(' ', n=1, expand=True)
        means[0] = means[0].astype(float)
        if larger_is_better_dict[col] and not small_is_better:
            top_n = means[0].nlargest(2).index.tolist()
        else:
            top_n = means[0].nsmallest(2).index.tolist()
        rest = list(df[col].index)
        for i, best in enumerate(top_n):
            df[col][best] = formats[i][0] + f'{df[col][best]}' + formats[i][1]
            rest.remove(best)

    return df


def aggregate_metrics(df, larger_is_better):
    ranks = []

    for col in df.columns:
        # as pm formatting occured before, extract means:
        print(df[col])
        means = df[col].str.split(' ', n=1, expand=True)
        means[0] = means[0].astype(float)

        # descending: biggest is first
        # right now: smallest wins
        if larger_is_better[col]:
            to_sort = - means[0]
        else:
            to_sort = means[0]

        to_sort = to_sort.to_numpy()
        order = to_sort.argsort()
        rank = order.argsort().astype("float")

        rank[np.isnan(to_sort)] = 10

        rank = rank + 1

        # rank[~np.isnan(to_sort)] = rank[~np.isnan(to_sort)] + 1
        # rank_numpy = rank.to_numpy()
        # rank_numpy[rank_numpy == 11] = float("NaN")

        rank[rank == 11] = float("NaN")

        ranks.append(rank)

        # rankings.append(ranking)

    # rankings = np.array(rankings)
    ranks = np.array(ranks)

    return ranks


def nested_dict():
    return collections.defaultdict(nested_dict)


# CAVE: set the following paths accordingly!
outpath = 'tex_Geom'
path_comp = 'experiments/train_model/evaluation/repetitions'
path_ae = 'experiments/fit_competitor/evaluation/repetitions'

Path(outpath).mkdir(exist_ok=True)

# files_ae = glob.glob(os.path.join(os.path.dirname(__file__), "..", path_ae, "/**/run.json"), recursive=True)
files_ae = glob.glob(path_ae + '/**/run.json', recursive=True)
# files_comp = glob.glob(os.path.join(os.path.dirname(__file__), "..", path_comp, "/**/run.json"), recursive=True)
files_comp = glob.glob(path_comp + '/**/run.json', recursive=True)
filelist = files_ae + files_comp

# use non aggregated, i.e. with MSE
used_measures = ['kl_global_01', 'kl_global_100', 'knn_recall', 'rmse', 'mean_trustworthiness',
                 'spearman_metric', '_mse', 'reconstruction']

# use for aggregated, i.e. without MSE
# used_measures = ['kl_global_01', 'kl_global_100', 'knn_recall', 'rmse', 'mean_trustworthiness',
#                'spearman_metric']

# mean_neighbourhood_loss, mean_continuity, mean_rank_correlation, mean_mrre, stress, density_global

# list of flat dicts
results = []
experiment = nested_dict()  # defaultdict(dict)
experiment_stats = nested_dict()

# 1. Gather all results in experiment dict
datasets = []
models = []
repetitions = np.arange(1, 6)
all_used_keys = []

ignored_models = ["ParametricUMAP_old"]
ignored_datasets = ["Zilionis", "FashionMNIST_old", "Earth", "MNIST_optimized", "Images", "PBMC"]

for filename in filelist:
    split = filename.split('/')
    repetition = int(split[-4][-1])
    dataset = split[-3]  # TODO: change to -3 and -2 for real results!
    if dataset not in datasets and dataset not in ignored_datasets:
        datasets.append(dataset)
    model = split[-2]

    if model in ignored_models or dataset in ignored_datasets:
        continue

    # nice name for proposed method:
    if 'LinearAE' in model:
        model = 'TopoPCA'
    elif 'TopoRegEdge' in model:
        model = 'TopoAE'
    elif 'GeomReg' in model:
        model = 'GeomAE'
    elif 'vanilla' in model:
        model = 'VanillaAE'
    elif 'UMAP_default' == model:
        model = "UMAP (default)"
    elif 'TSNE_default' in model:
        model = "TSNE (default)"
    elif 'PCA_default' in model:
        model = "PCA (default)"
    elif 'ParametricUMAP_default' == model:
        model = "Parametric UMAP (default)"
    if model not in models:
        models.append(model)

    with open(filename, 'rb') as f:
        data = json.load(f)

    run_file = filename.split('/')[-1]
    metrics_path = filename.strip(run_file) + 'metrics.json'
    with open(metrics_path, 'rb') as f:
        metrics = json.load(f)
    # metrics['testing.reconstruction_error']['values'][-1] for accessing latest recon error..

    if 'result' not in data.keys():
        continue

    result_keys = list(data['result'].keys())

    # used_keys = [key for key in result_keys if 'test_density_kl_global_' in key]
    used_keys = [key for key in result_keys if any([measure in key for measure in used_measures])]

    # Update list of all keys ever used for later processing (use more speaking name for PCA test recon
    for key in used_keys:
        if key == 'test_mse':
            new_key = 'test.reconstruction_error'  # better name for PCA test reconstruction error
        else:
            new_key = key
        if new_key not in all_used_keys:
            all_used_keys.append(new_key)

    # fill eval measures into experiments dict:
    for key in used_keys:  # still loop over old naming, as it is stored this way in json..
        if key in ["test_knn_recall", "test_density_kl_global_1000", "test_density_kl_global_10000"]:
            continue
        wrong_reconstruction = False
        if key in data['result'].keys():
            if key == 'test_mse':
                new_key = 'test.reconstruction_error'  # better name for PCA test reconstruction error
                if model in ['TSNE', 'UMAP']:  # "GeomAE (proposed)":  # in ['TSNE', 'UMAP']:
                    wrong_reconstruction = True
            else:
                new_key = key
            if key == 'training.reconstruction_error':  # use test recon (stored in metrics)
                new_key = 'test.reconstruction_error'
                if new_key not in all_used_keys:
                    all_used_keys.append(new_key)
                test_recon = metrics['testing.reconstruction_error']['values'][-1]
                print(f'Entering into {new_key} following Test Recon: {test_recon} for {dataset}/{model}/{repetition}')
                experiment[dataset][model][new_key][repetition] = test_recon
            else:
                if not wrong_reconstruction:
                    experiment[dataset][model][new_key][repetition] = data['result'][key]["value"]
                else:
                    experiment[dataset][model][new_key][repetition] = float("NaN")

# 2. Check that 5 repes avail and then compute mean + std
for dataset in datasets:
    if dataset in ignored_datasets:
        continue

    for model in models:
        loaded_results = experiment[dataset][model]
        for key in all_used_keys:
            if key in loaded_results.keys():  # not all methods have recon error
                rep_vals = np.array(list(loaded_results[key].values()))
                n_reps = len(rep_vals)
                if n_reps < 5:
                    print(f'Less than 5 reps in exp: {dataset}/{model}/{key}')
                    # embed()
                else:
                    # write mean and std into exp dict:
                    experiment[dataset][model][key]['mean'] = rep_vals.mean()
                    experiment[dataset][model][key]['std'] = rep_vals.std()
                    # Format mean +- std in experiment_stats dict
                    mean = rep_vals.mean()
                    std = rep_vals.std()

                    if not math.isnan(mean) and not math.isnan(std):
                        experiment_stats[dataset][model][key] = round_significant([mean], [std])[0]

                    else:
                        experiment_stats[dataset][model][key] = float("nan")

                    # if 'test_density_kl_global_100' in key:
                    #    experiment_stats[dataset][model][key] = round_significant([mean], [std])[
                    #        0]  # = f'{mean:1.12f}' + ' $\pm$ ' + f'{std:1.12f}'
                    #    # experiment_stats[dataset][model][key] = f'{mean:1.10f}'
                    # else:
                    #    experiment_stats[dataset][model][key] = round_significant([mean], [std])[
                    #        0]  # f'{mean:1.5f}' + ' $\pm$ ' + f'{std:1.5f}'
                    #    # experiment_stats[dataset][model][key] = f'{mean:1.5f}'

col_mapping = {'test_density_kl_global_0001': '$\dkl_{0.001}$',
               'test_density_kl_global_001': '$\dkl_{0.01}$',
               'test_density_kl_global_01': '$\dkl_{0.1}$',
               'test_density_kl_global_1': '$\dkl_{1}$',
               'test_density_kl_global_10': '$\dkl_{10}$',
               'test_density_kl_global_20': '$\dkl_{20}$',
               'test_density_kl_global_50': '$\dkl_{50}$',
               'test_density_kl_global_100': '$\dkl_{100}$',
               'test_density_kl_global_500': '$\dkl_{500}$',
               'test_density_kl_global_1000': '$\dkl_{1000}$',
               'test_density_kl_global_10000': '$\dkl_{10000}$',
               'test_mean_continuity': '$\ell$-Cont',
               'test_mean_mrre': '$\ell$-MRRE',
               'test_mean_trustworthiness': '$\ell$-Trust',
               'test_rmse': '$\ell$-RMSE',
               'test.reconstruction_error': 'MSE',
               'test_mean_rank_correlation': '$\\text{Corr}_{\\text{Rank}}$',
               'test_mean_neighbourhood_loss': '$\ell$-Nbhd',
               'test_density_global': '$\\text{D}_{\\text{glob}}$',
               'test_stress': 'Stress',
               'test_knn_recall': 'knn recall',
               'test_mean_knn_recall': 'kNN',
               'test_spearman_metric': 'Spear',
               }

# use for non aggregated, i.e. with MSE
order_measures = ['$\\dkl_{0.1}$', 'kNN', '$\ell$-Trust', '$\ell$-RMSE', '$\\dkl_{100}$', 'Spear', 'MSE']
# use for aggregated, i.e. without MSE
# order_measures = ['$\\dkl_{0.1}$', 'kNN', '$\ell$-Trust', '$\ell$-RMSE', '$\\dkl_{100}$', 'Spear']


larger_is_better = {
    '$\dkl_{0.001}$': 0,
    '$\dkl_{0.01}$': 0,
    '$\dkl_{0.1}$': 0,
    '$\dkl_{1}$': 0,
    '$\dkl_{10}$': 0,
    '$\dkl_{20}$': 0,
    '$\dkl_{50}$': 0,
    '$\dkl_{100}$': 0,
    '$\dkl_{500}$': 0,
    '$\dkl_{1000}$': 0,
    '$\dkl_{10000}$': 0,
    '$\ell$-Cont': 1,
    '$\ell$-MRRE': 0,
    '$\ell$-Trust': 1,
    '$\ell$-RMSE': 0,
    'MSE': 0,
    '$\\text{Corr}_{\\text{Rank}}$': 1,
    '$\ell$-Nbhd': 0,
    'knn recall': 1,
    'kNN': 1,
    '$\\text{D}_{\\text{glob}}$': 0,
    'Stress': 0,
    '$\langle \\text{rank} \rangle$': 0,
    '$\langle \\text{rank*} \rangle$': 0,
    'Spear': 1,
}

rankings = []

for dataset in datasets:
    print(dataset)
    # if dataset == "Zilionis":
    #    continue

    df = pd.DataFrame.from_dict(experiment_stats[dataset], orient='index')
    df = df.rename(columns=col_mapping)

    # df['order'] = [7, 6, 5, 2, 3, 1]
    df['order'] = [7, 6, 5, 4, 2, 3, 1]

    # columns = np.delete(columns, np.where(columns == "order"))

    df = df.sort_values(by=['order'])
    df = df.drop(columns=['order'])

    df = df.reindex(order_measures, axis=1)
    ranking = aggregate_metrics(df, larger_is_better)
    rankings.append(ranking)

    df = highlight_best_with_std(df, larger_is_better)

    # df = df.reindex(order_measures, axis=1)
    # df = df[order_measures]

    with pd.option_context("max_colwidth", 10000):
        df.to_latex(f'{outpath}/{dataset}_table_5_digits.tex', escape=False)

rankings = np.stack(rankings)

# calculate mean over all datasets
mean_ranking = np.mean(rankings, axis=0).T

# calculate mean over all metrics, for a fixed model
overall_ranks = np.mean(mean_ranking, axis=1).astype("float")

df["$\langle \\text{rank} \rangle$"] = overall_ranks

mean_mean_ranking_with_mse = np.expand_dims(overall_ranks, axis=1).round(3).astype(str)
data = np.hstack((mean_ranking, mean_mean_ranking_with_mse))

columns = df.columns.values
rows = df.index.values

df = pd.DataFrame(data, columns=columns, index=rows)
df = highlight_best_with_std(df, larger_is_better, small_is_better=True)

# df = df.reindex(order_measures_ranks, axis=1)
# df = df[order_measures_ranks]

with pd.option_context("max_colwidth", 10000):
    df.to_latex(f'{outpath}/ranks_aggregated_3_digits.tex', escape=False)

# convert to df: df = pd.DataFrame.from_dict(experiment, orient='index')
# format is then df['Vanilla']['CIFAR']['test_mean_mrre']['mean']


"""# remove column index of data mse
idx = np.arange(rankings.shape[1])
idx = np.delete(idx, np.where(df.columns.values == "MSE"))

rankings_without_mse = rankings[:, idx]
rankings_with_mse = rankings

# calculate mean over all datasets
mean_ranking_with_mse = np.nanmean(rankings, axis=0).T
mean_ranking_without_mse = np.nanmean(rankings_without_mse, axis=0).T

# replace 11. by NaN
# mean_ranking_with_mse[mean_ranking_with_mse == 11.] = float("NaN")
# mean_ranking_with_mse = mean_ranking_with_mse.round(3).astype(str)

# TODO: make sure I calc the mean mean ranking without mse without mse for all methods!
# calculate mean over all metrics
mean_mean_ranking_without_mse = np.nanmean(mean_ranking_without_mse, axis=1).astype("float")
mean_mean_ranking_with_mse = np.nanmean(mean_ranking_with_mse, axis=1).astype("float")

# mean_ranking_without_mse = mean_ranking_without_mse.round(3).astype(str)
# mean_ranking_with_mse = mean_ranking_with_mse.round(3).astype(str)

df["$\langle \\text{rank*} \rangle$"] = mean_mean_ranking_without_mse
df["$\langle \\text{rank} \rangle$"] = mean_mean_ranking_with_mse

mean_mean_ranking_without_mse = np.expand_dims(mean_mean_ranking_without_mse, axis=1).round(3).astype(str)
mean_mean_ranking_with_mse = np.expand_dims(mean_mean_ranking_with_mse, axis=1).round(3).astype(str)
mean_ranking = np.hstack((mean_ranking_with_mse, mean_mean_ranking_without_mse, mean_mean_ranking_with_mse))

columns = df.columns.values
rows = df.index.values

df = pd.DataFrame(mean_ranking, columns=columns, index=rows)
df = highlight_best_with_std(df, larger_is_better, small_is_better=True)

# df = df.reindex(order_measures_ranks, axis=1)
# df = df[order_measures_ranks]

with pd.option_context("max_colwidth", 10000):
    df.to_latex(f'{outpath}/ranks_aggregated_3_digits.tex', escape=False)

# convert to df: df = pd.DataFrame.from_dict(experiment, orient='index')
# format is then df['Vanilla']['CIFAR']['test_mean_mrre']['mean']
"""
