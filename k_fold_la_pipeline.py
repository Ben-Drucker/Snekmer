# %%

import collections

### Imports
import copy
import functools
import io
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import random
import re
import requests
import shutil
import subprocess
import sys
import time
import tqdm
import typing
import warnings
import yaml
from matplotlib import figure, rcParams
from scipy import stats

# %%
from sklearn.model_selection import StratifiedKFold

rcParams["font.family"] = "Fira Code"
# rcParams["font.family"] = "Palatino"
rcParams["axes.titleweight"] = "bold"
rcParams["axes.labelsize"] = "large"

### Copy and Paste Tools
# Skip cell: %%script false --no-raise-error

num_pos = 6664
num_neg = 106757

pd.DataFrame(
    {
        "id": [f"Positive_{i}" for i in range(num_pos)] + [f"Negative_{i}" for i in range(num_neg)],
        "Family": ["POS"] * num_pos + ["NEG"] * num_neg,
    }
).to_csv("skf/annotations/annots.ann", index=False, sep="\t")

# %%
input_seq_id_to_seq: dict[str, str] = {}
for filename in [
    "/people/druc594/containers/python-tunneller.sif/home/tunneller/Snekmer/execution/input/all_pos_6664.fasta",
    "/people/druc594/containers/python-tunneller.sif/home/tunneller/Snekmer/execution/input/all_neg_106757.fasta",
]:
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(">"):
                input_seq_id = line.strip().split(">")[-1]
            else:
                input_seq = line.strip()
                input_seq_id_to_seq[input_seq_id] = input_seq

X = np.array(list(input_seq_id_to_seq.values()))
X_lab = np.array(list(input_seq_id_to_seq.keys()))
y = np.array([0 if "neg" in seq_id.lower() else 1 for seq_id in input_seq_id_to_seq.keys()])

# %%
assert X.shape[0] == y.shape[0]


os.chdir("/people/druc594/containers/python-tunneller.sif/home/tunneller/Snekmer/execution")
N_FOLDS = 3

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

inputs_dir = "skf/input"
if os.path.exists(inputs_dir):
    shutil.rmtree(inputs_dir)
os.makedirs(inputs_dir)

annotation_dir = "skf/annotations"
num_neg = -1
num_pos = -1
info_str_short = ""
for fold, (train_indexes, test_indexes) in tqdm.tqdm(enumerate(skf.split(X, y)), total=N_FOLDS):
    for tr_te, idx_set in zip(["train", "test"], [train_indexes, test_indexes]):
        for class_, cls_ in zip([0, 1], ["Negative", "Positive"]):
            temp_filename = os.path.join(inputs_dir, "temp.fasta")
            with open(temp_filename, "w") as f:
                n_seqs = 0
                for idx in idx_set:
                    if y[idx] == class_:
                        f.write(f">|{X_lab[idx]}|\n{X[idx]}\n")
                        n_seqs += 1

            info_str_short = f"f{fold}trte{tr_te}"
            info_str = f"{cls_}_{n_seqs}"
            final_filename = os.path.join(inputs_dir, f"{info_str}.{info_str_short}")
            os.rename(temp_filename, final_filename)

            if cls_ == "Positive":
                num_pos = n_seqs

                with open("skf/base_config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                    config["input_file_exts"] = [f"{info_str_short}"]
                    config["input_file_regex"] = ".*"
                with open(os.path.join("skf/configs", f"{info_str_short}.yaml"), "w") as f:
                    yaml.dump(config, f)

            else:
                num_neg = n_seqs

# %%
os.chdir("/people/druc594/containers/python-tunneller.sif/home/tunneller/Snekmer/execution/skf")


def copy_folders(sub_name: str):
    shutil.copytree(
        os.path.join("output", "apply_inputs", sub_name),
        os.path.join(sub_name),
        dirs_exist_ok=True,
    )


result_dfs = {}
for fold in tqdm.tqdm(range(N_FOLDS)):
    train_cmd = f"~/.local/bin/snekmer learn --forcerun --configfile configs/f{fold}trtetrain.yaml"
    print(train_cmd)
    subprocess.run(train_cmd, shell=True)

    [copy_folders(sub_name) for sub_name in ["confidence", "counts", "stats"]]

    test_cmd = f"~/.local/bin/snekmer apply --forcerun --configfile configs/f{fold}trtetest.yaml"
    print(test_cmd)
    p = subprocess.run(test_cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError(f"Error in fold {fold}: {p.stderr.decode('utf-8')}")

    result_df = {
        os.path.join("output", "apply", x): pd.read_csv(os.path.join("output", "apply", x))
        for x in os.listdir("output/apply")
        if x.endswith(".csv")
    }

    result_dfs |= {fold: result_df}
    try:
        shutil.rmtree(os.path.join("output"))
    except Exception as e:
        print(f"Error removing output folder: {e}")


with open("result_dfs.pkl", "wb") as f:
    pickle.dump(result_dfs, f)