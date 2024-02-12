import torch
import sys
sys.path.append("./")
from load import load_fine_definitions
from transformers import AutoTokenizer

DEBUG = 0

# PATHS
DATA_PATH           = 'data'
COARSE_DATA_PATH    = DATA_PATH + '/coarse-grained/'
FINE_DATA_PATH      = DATA_PATH + '/fine-grained/'
MAP_PATH            = 'hw2/data/coarse_fine_defs_map.json'

# DATA
COARSE_TRAIN_DATA   = COARSE_DATA_PATH + 'train_coarse_grained.json'
COARSE_VAL_DATA     = COARSE_DATA_PATH + '/val_coarse_grained.json'
COARSE_TEST_DATA    = COARSE_DATA_PATH + '/test_coarse_grained.json'

FINE_TRAIN_DATA     = FINE_DATA_PATH + '/train_fine_grained.json'
FINE_VAL_DATA       = FINE_DATA_PATH + '/val_fine_grained.json'
FINE_TEST_DATA      = FINE_DATA_PATH + '/test_fine_grained.json'

PREDICTION_PATH     = DATA_PATH + '/predictions/prediction.csv'

SAVE_MODEL_PATH     = 'model'
LOAD_MODEL_PATH     = SAVE_MODEL_PATH + '/cls-token-ws.ckpt'


# MODEL
MODEL_NAME          = "bert-base-cased" 
tokenizer           = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False) 
BATCH_SIZE          = 16
WS                  = True
MODE                = "CLS"  # [CLS, TARGET]
DROPOUT             = 0.1
EPOCHS              = 30
LR                  = 1e-5
WEIGHT_DECAY        = 0.01
NUM_CLASSES         = 2      # YES, NO
SEED                = 42

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"





definitions, coarse_to_fine     = load_fine_definitions(MAP_PATH)
sen_to_idx                      = {sen: idx for idx, sen in enumerate(definitions.keys())}
fine_to_coarse                  = {fine: coarse for coarse, fines in coarse_to_fine.items() for fine in fines}

pos_map                         = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 's': 'ADJ'}
single_label_pairs_fine         = {}
single_label_pairs_course       = {}

label_pairs_fine                = {}
label_pairs_course              = {}

num_classes_fine                = len(sen_to_idx)

