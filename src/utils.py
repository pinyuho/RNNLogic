import sys
import os
import logging
import argparse
import random
import json
import yaml
import easydict
import numpy as np
import torch
import comm

from torch.utils.data import Dataset, Subset

def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()

    if "---" in raw_text:
        configs = []
        grid, template = raw_text.split("---")
        grid = yaml.safe_load(grid)
        template = jinja2.Template(template)
        for hyperparam in meshgrid(grid):
            config = easydict.EasyDict(yaml.safe_load(template.render(hyperparam)))
            configs.append(config)
    else:
        configs = [easydict.EasyDict(yaml.safe_load(raw_text))]

    return configs

def save_config(cfg, path):
    with open(os.path.join(path, 'config.yaml'), 'w') as fo:
        yaml.dump(dict(cfg), fo)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

def save_model(model, optim, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    params = {
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }

    torch.save(params, os.path.join(args.save_path, 'checkpoint'))

def load_model(model, optim, args):
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])

def set_logger(save_path):
    log_file = os.path.join(save_path, 'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class SubsetProxy(Dataset):
    """
    一个既有 Subset 功能，又能把原 dataset
    的所有属性和方法透明代理过来的包装器。
    """
    def __init__(self, dataset: Dataset, indices):
        self._dataset = dataset
        self._indices = list(indices)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]

    def __getattr__(self, name):
        # 只要这个 proxy 本身没有的属性，就去底层 dataset 拿
        return getattr(self._dataset, name)
    
def get_subset_dataset(dataset, ratio: float = 1.0, seed: int = 42):
    """
    ratio < 1.0  → 回传 SubsetProxy(dataset, indices)
    ratio >= 1.0 → 回传原 dataset
    """
    if ratio >= 1.0:
        return dataset

    n = len(dataset)
    k = int(n * ratio)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, k, replace=False)

    if comm.get_rank() == 0:
        logging.info(f"[Subset] {dataset.__class__.__name__}: {k}/{n} samples ({ratio:.0%})")
    return SubsetProxy(dataset, idx)

def build_rotate_embedding(path, num_extra=2, scale=0.01):
    """
    讀 RotatE 向量，再補 2 格隨機權重 ⇒ nn.Embedding
    """
    base = torch.as_tensor(np.load(path), dtype=torch.float)          # (R, D)
    extra = torch.randn(num_extra, base.size(1)).mul_(scale)          # (2, D)
    weight = torch.cat([base, extra], dim=0)                          # (R+2, D)
    return torch.nn.Embedding.from_pretrained(
        weight, freeze=True, padding_idx=weight.size(0)-1             # 最後一格當 padding
    )
