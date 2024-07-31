import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys

sys.path.append(os.getcwd())
from data.AttriDataset import AttriDataset
from torch.utils.data import WeightedRandomSampler
import numpy as np
from config import read_yaml
import pdb
from tqdm import tqdm
from collections import Counter


def balance_data(dataset, cfg):
    weights = []
    labels = dataset.label_list
    label_counter = Counter(labels)
    count = len(labels)
    nclasses = cfg.model.num_class
    weight_per_class = [0.0] * nclasses
    weight = [0] * len(labels)
    N = float(count)
    for i in range(nclasses):
        weight_per_class[i] = N / float(label_counter[i])
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    weights.append(weight)

    # 对所有weight相乘
    weight = np.array([1] * len(labels))
    for w in weights:
        weight = np.multiply(weight, np.array(w))

    return weight


def build_dataloader(cfg, data_type):
    dataset = AttriDataset(cfg, data_type)
    weights = balance_data(dataset, cfg)
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, cfg.train.batch_size, num_workers=cfg.train.num_workers, drop_last=True
    )
    return dataloader
