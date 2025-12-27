# src/defectvad/utils.py

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)