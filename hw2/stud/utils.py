import sys
sys.path.append("./hw2/stud/")
import torch
import config
import random
import numpy as np


def seed_everything(seed = config.SEED):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True