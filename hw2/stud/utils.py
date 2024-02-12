import sys
sys.path.append("./hw2/stud/")
import torch
import config
import random
import numpy as np
import torch.nn.functional as F


def seed_everything(seed: int = 42): 
    """
    Seed everything for reproducibility

    Args:
    - seed: the seed to use for reproducibility

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def glossBERT_collate_fn(batch: list):
    """
    Collate function for the glossBERT dataset, pads the input sequences to the same length
    """

    instance_ids, input_ids, candidates, attention_masks, target_masks, token_type_ids, labels = zip(*batch) 

    # max length of the input sequences in the batch for padding
    max_len = max([len(input_sequence) for input_sequence in input_ids])

    # pad the input sequences
    input_ids = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in input_ids])
    token_type_ids = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in token_type_ids])
    target_masks = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in target_masks])
    attention_masks = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in attention_masks])
    labels = torch.tensor(labels)

    return instance_ids, input_ids, candidates, attention_masks, target_masks, token_type_ids, labels

    
def generate_glossBERT_pairs(sentence: list,
                            candidates: dict,
                            target_idx: int,
                            sense: str,
                            pos_tag: str,
                            instance_id: str,
                            ws: bool = True) -> list:
    """
    Generate the glossBERT pairs to feed into the model
    
    Args:
    - sentence: a list of tokens representing the sentence
    - candidates: a dictionary containing the candidate senses for the target word
    - target_idx: the index of the target word
    - sense: the correct sense for the target word
    - pos_tag: the pos tag of the target word
    - instance_id: the instance id of the sentence
    - ws: a boolean indicating whether to use the weak supervision or not
    """

    target_word = sentence[target_idx]

    if ws:
        # wrap the target word with quotes and shift the target index
        base_sentence = sentence[:target_idx] + ["\""] + [sentence[target_idx]] + ["\""] + sentence[target_idx+1:]
        target_idx = target_idx + 1
    else:
        base_sentence = sentence[:target_idx] + [sentence[target_idx]] + sentence[target_idx+1:]

    all_pairs = []

    for candidate in candidates:

        candidate_pos = candidate.split(".")[1]

        # if the pos tag of the candidate is not the same as the pos tag of the target word, skip
        if config.pos_map[candidate_pos] != pos_tag:
            continue
        
        # load the definition of the candidate and construct the gloss
        definition = config.definitions[candidate]
        gloss = f"{target_word} : {definition}"

        # get the input ids, token type ids, target masks, and attention masks of the gloss pairs
        input_ids, token_type_ids, target_mask, attention_mask = _get_ids(base_sentence, gloss, target_idx)

        label = 1 if candidate == sense else 0

        # set the sense labels for the instance if the gloss is the correct sense
        if label == 1:
            config.single_label_pairs_fine[instance_id] = config.fine_to_coarse[candidate]

        all_pairs.append((input_ids, token_type_ids, target_mask, attention_mask, label, instance_id, candidate))

    return all_pairs

        


def _get_ids(sentence: list, gloss: str, target_idx: int) -> tuple:
    """
    Get the input ids, token type ids, target masks, and attention masks of the gloss pairs

    Args:
    - sentence: a list of tokens representing the sentence
    - gloss: a string representing the gloss
    - target_idx: an integer representing the index of the target word
    """
    
    input_ids = [config.tokenizer.cls_token_id]
    position_ids = [-1]
    shifted_target_idx = -1

    # encode the sentence and the gloss and compute the position ids
    for i, token in enumerate(sentence):
        if i == target_idx:
            shifted_target_idx = len(input_ids)
        encoded_token = config.tokenizer.encode(token, add_special_tokens=False)
        input_ids.extend(encoded_token)
        position_ids.extend([i] * len(encoded_token))

    # add the [SEP] token to separate the sentence from the gloss
    input_ids.append(config.tokenizer.sep_token_id)
    position_ids.append(-1)
    token_type_ids = [0] * len(input_ids)

    # encode the gloss
    encoded_gloss = config.tokenizer.encode(gloss, add_special_tokens=False)

    # add the gloss to the input ids and extend the position ids and token type ids
    input_ids.extend(encoded_gloss)
    position_ids.extend([-1] * len(encoded_gloss))
    token_type_ids.extend([1] * len(encoded_gloss))
    attention_mask = [1] * len(input_ids)

    # create a mask for the target word
    target_mask = [0] * len(input_ids)
    target_mask[shifted_target_idx] = 1

    # transform the input ids, token type ids, and target mask to tensors
    input_ids = torch.tensor(input_ids)
    token_type_ids = torch.tensor(token_type_ids)
    target_mask = torch.tensor(target_mask).to(torch.bool)
    attention_mask = torch.tensor(attention_mask)

    return input_ids, token_type_ids, target_mask, attention_mask



