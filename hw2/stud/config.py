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

PREDICTION_PATH = '../../data/predictions/prediction.csv'

# INTERMEDIATE DATA
SENSE_EMBEDDINGS_PATH = '../../data/intermediate/sense_embeddings.npy'


# PADDINGS
PAD_TOKEN = "<PAD>"
CLS_TOKEN = "<s>" # "[CLS]" #
SEP_TOKEN = "</s>" # "[SEP]" #
DELIMITER_TOKEN = "\""

# MODEL
BATCH_SIZE = 16
WS = True



SEED = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"





definitions, coarse_to_grain = load_fine_definitions(MAP_PATH)
sen_to_idx = {sen: idx for idx, sen in enumerate(definitions.keys())}
grain_to_coarse = {grain: coarse for coarse, grain in coarse_to_grain.items()}

pos_map = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 's': 'ADJ'}
label_pairs_fine = {}
label_pairs_course = {}


num_classes_fine = len(sen_to_idx)
# idx_to_emb = {idx: sentence_embeddings(definitions[sen]) for sen, idx in sen_to_idx.items()}