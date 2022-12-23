import os
from io import BytesIO
import subprocess
from datetime import datetime

import torch

import pandas as pd

from torch.utils.tensorboard import SummaryWriter


def get_least_busy_gpu(verbose=True):
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    gpu_df["memory.free"] = pd.to_numeric(gpu_df["memory.free"])
    idx = gpu_df['memory.free'].idxmax()

    if verbose:
        print('GPU usage:\n{}'.format(gpu_df))
        print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))

    return idx


# determine least busy device
least_busy_device = get_least_busy_gpu(verbose=True)

device = torch.device(f"cuda:{least_busy_device}" if torch.cuda.is_available() else "cpu")

# lower bound for numerical stability
LOWER_EPSILON = 1e-20
BIGGER_LOWER_EPSILON = 1e-12
BIGGEST_LOWER_EPSILON = 1e-10
UPPER_EPSILON = 1e20
SMALLER_UPPER_EPSILON = 1e12


def init():
    global topomodel
    topomodel = False

output_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output"

def get_logdir(subdir):
    if subdir:
        return os.path.join(output_path, f"runs/{datetime.now().strftime('%Y.%m.%d')}/{subdir}/{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")
    else:
        return os.path.join(output_path, f"runs/{datetime.now().strftime('%Y.%m.%d')}/{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")


def get_summary_writer(subdir=None):
    return SummaryWriter(get_logdir(subdir))
