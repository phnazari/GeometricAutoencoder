#!/bin/bash

## Reproduce our Results for MNIST

# train a geometric autoencoder
bash scripts/create_eval_configs.sh

# move the results to the proper place
mkdir -p experiments/train_model/evaluation/repetitions/rep1/MNIST/
mkdir -p experiments/train_model/evaluation/repetitions/rep2/MNIST/
mkdir -p experiments/train_model/evaluation/repetitions/rep3/MNIST/
mkdir -p experiments/train_model/evaluation/repetitions/rep4/MNIST/
mkdir -p experiments/train_model/evaluation/repetitions/rep5/MNIST/

mv save_config/1 experiments/train_model/evaluation/repetitions/rep1/MNIST/GeomReg
mv save_config/2 experiments/train_model/evaluation/repetitions/rep2/MNIST/GeomReg
mv save_config/3 experiments/train_model/evaluation/repetitions/rep3/MNIST/GeomReg
mv save_config/4 experiments/train_model/evaluation/repetitions/rep4/MNIST/GeomReg
mv save_config/5 experiments/train_model/evaluation/repetitions/rep5/MNIST/GeomReg

# run our indicatrix and determinant diagnostics
python3 exp/analysis.py

# evaluate the model quantitatively
python3 scripts/load_results.py
