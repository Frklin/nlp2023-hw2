import sys
sys.path.append("./hw2/stud/")
import torch
import config
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F



#print special tokens
#print(tokenizer.all_special_tokens)

def seed_everything(seed = 42): 
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def generate_glossBERT_pairs(sentence, candidates, target_idx, sense, pos_tag, instance_id, ws=True):
    
    target_word = sentence[target_idx]

    if ws:
        base_sentence = sentence[:target_idx] + ["\""] + [sentence[target_idx]] + ["\""] + sentence[target_idx+1:]
        target_idx = target_idx + 1
    else:
        base_sentence = sentence[:target_idx] + [sentence[target_idx]] + sentence[target_idx+1:]

    # base_sentence = " ".join(base_sentence)

    all_pairs = []

    for candidate in candidates:

        candidate_pos = candidate.split(".")[1]

        if config.pos_map[candidate_pos] != pos_tag:
            continue
        
        definition = config.definitions[candidate]
        gloss = f"{target_word} : {definition}"
        input_ids, token_type_ids, target_mask, attention_mask = _get_ids(base_sentence, gloss, target_idx)

        label = 1 if candidate == sense else 0

        if label == 1:
            config.single_label_pairs_fine[instance_id] = config.fine_to_coarse[candidate]

        all_pairs.append((input_ids, token_type_ids, target_mask, attention_mask, label, instance_id, candidate))

    return all_pairs

        


def _get_ids(sentence, gloss, target_idx):
    
    input_ids = [config.tokenizer.cls_token_id]
    position_ids = [-1]
    shifted_target_idx = -1

    for i, token in enumerate(sentence):
        if i == target_idx:
            shifted_target_idx = len(input_ids)
        encoded_token = config.tokenizer.encode(token, add_special_tokens=False)
        input_ids.extend(encoded_token)
        position_ids.extend([i] * len(encoded_token))

    input_ids.append(config.tokenizer.sep_token_id)
    position_ids.append(-1)
    token_type_ids = [0] * len(input_ids)

    encoded_gloss = config.tokenizer.encode(gloss, add_special_tokens=False)
    input_ids.extend(encoded_gloss)
    position_ids.extend([-1] * len(encoded_gloss))
    token_type_ids.extend([1] * len(encoded_gloss))
    attention_mask = [1] * len(input_ids)

    target_mask = [0] * len(input_ids)
    target_mask[shifted_target_idx] = 1

    input_ids = torch.tensor(input_ids)
    token_type_ids = torch.tensor(token_type_ids)
    target_mask = torch.tensor(target_mask).to(torch.bool)
    attention_mask = torch.tensor(attention_mask)

    return input_ids, token_type_ids, target_mask, attention_mask




def glossBERT_collate_fn(batch):
    instance_ids, input_ids, candidates, attention_masks, target_masks, token_type_ids, labels = zip(*batch) 
    max_len = max([len(input_sequence) for input_sequence in input_ids])

    input_ids = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in input_ids])
    token_type_ids = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in token_type_ids])
    target_masks = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in target_masks])
    attention_masks = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in attention_masks])
    labels = torch.tensor(labels)

    return instance_ids, input_ids, candidates, attention_masks, target_masks, token_type_ids, labels

