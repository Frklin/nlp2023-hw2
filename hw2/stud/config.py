import torch

# PATHS
COARSE_DATA_PATH = '../../data/data/coarse-grained/'
FINE_DATA_PATH = '../../data/data/fine-grained/'
MAP_PATH = '../../data/data/map/coarse_fine_defs_map.json'

# DATA
COARSE_TRAIN_DATA = COARSE_DATA_PATH + 'train_coarse_grained.json'
COARSE_VAL_DATA = COARSE_DATA_PATH + '/val_coarse_grained.json'
COARSE_TEST_DATA = COARSE_DATA_PATH + '/test_coarse_grained.json'

FINE_TRAIN_DATA = FINE_DATA_PATH + '/train_fine_grained.json'
FINE_VAL_DATA = FINE_DATA_PATH + '/val_fine_grained.json'
FINE_TEST_DATA = FINE_DATA_PATH + '/test_fine_grained.json'



# PADDINGS
PAD_TOKEN = "<PAD>"


# MODEL
BATCH_SIZE = 32




SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
