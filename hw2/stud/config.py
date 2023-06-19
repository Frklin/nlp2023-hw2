import torch

# PATHS
COARSE_DATA_PATH = 'data/coarse'
FINE_DATA_PATH = 'data/fine'
MAP_PATH = 'data/map.json'

# DATA
COARSE_TRAIN_DATA = 'data/coarse/train.json'
COARSE_VAL_DATA = 'data/coarse/dev.json'
COARSE_TEST_DATA = 'data/coarse/test.json'

FINE_TRAIN_DATA = 'data/fine/train.json'
FINE_VAL_DATA = 'data/fine/dev.json'
FINE_TEST_DATA = 'data/fine/test.json'


# PADDINGS
PAD_TOKEN = "<PAD>"







DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
