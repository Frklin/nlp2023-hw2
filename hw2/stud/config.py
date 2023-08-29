import torch
import sys
sys.path.append("./")
from load import load_fine_definitions

# PATHS
COARSE_DATA_PATH = '../../data/coarse-grained/'
FINE_DATA_PATH = '../../data/fine-grained/'
MAP_PATH = '../../data/map/coarse_fine_defs_map.json'

# DATA
COARSE_TRAIN_DATA = COARSE_DATA_PATH + 'train_coarse_grained.json'
COARSE_VAL_DATA = COARSE_DATA_PATH + '/val_coarse_grained.json'
COARSE_TEST_DATA = COARSE_DATA_PATH + '/test_coarse_grained.json'

FINE_TRAIN_DATA = FINE_DATA_PATH + '/train_fine_grained.json'
FINE_VAL_DATA = FINE_DATA_PATH + '/val_fine_grained.json'
FINE_TEST_DATA = FINE_DATA_PATH + '/test_fine_grained.json'

# INTERMEDIATE DATA
SENSE_EMBEDDINGS_PATH = '../../data/intermediate/sense_embeddings.npy'


# PADDINGS
PAD_TOKEN = "<PAD>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
DELIMITER_TOKEN = "\""

# MODEL
BATCH_SIZE = 8




SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"





definitions = load_fine_definitions(MAP_PATH)
sen_to_idx = {sen: idx for idx, sen in enumerate(definitions.keys())}

num_classes_fine = len(sen_to_idx)
# idx_to_emb = {idx: sentence_embeddings(definitions[sen]) for sen, idx in sen_to_idx.items()}