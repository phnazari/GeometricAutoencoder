#!/bin/bash

###
# THIS FILE WAS TAKEN FROM https://github.com/BorgwardtLab/topological-autoencoders
###

# Autoencoder based
output_pattern='experiments/train_model/repetitions/{rep}/{config}'
# Competitor
output_pattern_competitor='experiments/fit_competitor/repetitions/{rep}/{config}'

python3 scripts/configs_from_product.py exp.train_model \
  --name config \
  --set \
  experiments/train_model/best_runs/MNIST/VanillaConv.json \
  experiments/train_model/best_runs/MNIST/GeomRegConv.json \
  experiments/train_model/best_runs/FashionMNIST/VanillaConv.json \
  experiments/train_model/best_runs/FashionMNIST/GeomRegConv.json \
  --name rep --set rep1 rep2 rep3 rep4 rep5 \
  --name dummy --set evaluation.active=True \
  --name dummy2 --set evaluation.evaluate_on='test' \
  --output-pattern ${output_pattern}

#python3 scripts/configs_from_product.py exp.fit_competitor \
#  --name config \
#  --set \
#  experiments/fit_competitor/best_runs/MNIST/ParametricUMAP_default.json \
#  experiments/fit_competitor/best_runs/FashionMNIST/ParametricUMAP_default.json \
#  experiments/fit_competitor/best_runs/PBMC_new/ParametricUMAP_default.json \
#  experiments/fit_competitor/best_runs/Zilionis/ParametricUMAP_default.json \
#  experiments/fit_competitor/best_runs/CElegans/ParametricUMAP_default.json \
#  --name rep --set rep1 rep2 rep3 rep4 rep5 \
#  --name dummy --set evaluation.active=True \
#  --name dummy2 --set evaluation.evaluate_on='test' \
#  --output-pattern ${output_pattern_competitor}

