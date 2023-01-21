# Geometric Autoencoder

This repository provides the code for the paper "Geometric Autoencoders: What You See is What You Decode"

## Getting Started

Clone the Repository

```
git clone https://github.com/Kingrimursel/GeometricAutoencoder.git
```

Create a Conda Environment

```
conda create --name my_environment
```

Change into the project directory and install the dependencies

```
cd GeometricAutoencoder
pip3 install -r requirements.txt
```

While TorchVision takes care of the MNIST and FashionMNIST datasets, you will have to [download](http://cb.csail.mit.edu/cb/densvis/datasets/ ) the PBMC, Zilionis and CElegans datasets yourself.

## Reproducing the Results

### TL; DR
If you want to reproduce our results for MNIST, you can do so by executing

```
bash main.sh
```

### More Detailed

What happens in this case, is that you first train a Geometric Autoencoder by executing
```
bash scripts/create/eval/configs.sh.
```
The training results will be placed in a folder called "save_config", and you will have to move them to the `experiments` folder in order to proceed with the evaluation. The first run, for example, should be moved like

```
mv save_config/1 experiments/train_model/evaluation/repetitions/rep1/MNIST/GeomReg
```

You can then run our geometric diagnostics by envoking
```
python3 exp/analysis.py
```
and the quantitative metrics by executing
```
python3 scripts/load_results.py
```


## The Differential Geometry
The differential geometry can be found inside directory `src/diffgeo`. Our geometric regularizer is implemented in `src/criterions.py`.

