#!/bin/bash

#ae_models=(Vanilla TopoRegEdgeSymmetric)
#competitor_methods=(PCA TSNE Isomap UMAP)
#output_pattern_ae='experiments/evaluate_model/dimensionality_reduction/{dataset}/{model}.json'
#output_pattern_competitor='experiments/fit_competitor/dimensionality_reduction/{dataset}/{model}.json'

# Autoencoder based
output_pattern='experiments/train_model/repetitions/{rep}/{config}'
# Competitor
output_pattern_competitor='experiments/fit_competitor/repetitions/{rep}/{config}'

#python3 scripts/configs_from_product.py exp.train_model \
#  --name config \
#  --set \
#  experiments/train_model/best_runs/MNIST/Vanilla.json \
#  experiments/train_model/best_runs/MNIST/TopoRegEdgeSymmetric.json \
#  experiments/train_model/best_runs/MNIST/GeomReg.json \
#  --name rep --set rep1 rep2 rep3 rep4 rep5 \
#  --name dummy --set evaluation.active=False \
#  --name dummy2 --set evaluation.evaluate_on='test' \
#  --output-pattern ${output_pattern}

#python3 scripts/configs_from_product.py exp.train_model \
#  --name config \
#  --set \
#  experiments/train_model/best_runs/PBMC_new/Vanilla.json \
#  experiments/train_model/best_runs/PBMC_new/TopoRegEdgeSymmetric.json \
#  experiments/train_model/best_runs/PBMC_new/GeomReg.json \
#  --name rep --set rep1 rep2 rep3 rep4 rep5 \
#  --name dummy --set evaluation.active=True \
#  --name dummy2 --set evaluation.evaluate_on='test' \
#  --output-pattern ${output_pattern}

python3 scripts/configs_from_product.py exp.fit_competitor \
  --name config \
  --set \
  experiments/fit_competitor/best_runs/MNIST/ParametricUMAP_default.json \
  experiments/fit_competitor/best_runs/FashionMNIST/ParametricUMAP_default.json \
  experiments/fit_competitor/best_runs/PBMC_new/ParametricUMAP_default.json \
  experiments/fit_competitor/best_runs/Zilionis/ParametricUMAP_default.json \
  experiments/fit_competitor/best_runs/CElegans/ParametricUMAP_default.json \
  --name rep --set rep1 rep2 rep3 rep4 rep5 \
  --name dummy --set evaluation.active=True \
  --name dummy2 --set evaluation.evaluate_on='test' \
  --output-pattern ${output_pattern_competitor}

#python3 scripts/configs_from_product.py exp.fit_competitor \
#  --name config \
#  --set \
#  experiments/fit_competitor/best_runs/PBMC_new/ParametricUMAP_default.json \
#  --name rep --set rep1 rep2 rep3 rep4 \
#  --name dummy --set evaluation.active=False \
#  --name dummy2 --set evaluation.evaluate_on='test' \
#  --output-pattern ${output_pattern_competitor}
