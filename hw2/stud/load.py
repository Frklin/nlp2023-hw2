import sys
sys.path.append("./")
sys.path.append("./hw2/stud/")
from torch.utils.data import Dataset
import json
import config
from utils import generate_glossBERT_pairs


class FineGrainedDataset(Dataset):
    """
    Dataset for fine-grained sense disambiguation
    
    Args:
    file_path (str): path to the json file containing the dataset
    """

    def __init__(self, file_path: str):
        self.data = self.load_data(file_path)

    def load_data(self, file_path: str):
        entries = []
        i = 0
        with open(file_path, 'r') as f:
            row = json.load(f)

        for idx, value in row.items():
            i += 1
            if config.DEBUG and i==100:
                break
            instance_ids = value["instance_ids"]
            words = value["words"]
            lemmas = value["lemmas"]
            pos_tags = value["pos_tags"]
            candidates = value["candidates"]
            senses = value["senses"]

            # set the sense labels for the metrics calculation
            set_sense_labels(idx,instance_ids, senses)

            # for each target word, for each candidate sense, generate a gloss pair
            for idx in candidates.keys():
                pos_tag = pos_tags[int(idx)]
                entries.extend(generate_glossBERT_pairs(words, candidates[idx], int(idx), senses[idx][0], pos_tag, instance_ids[idx], ws=config.WS))

        return entries
    
    def __getitem__(self, index: int):
        input_id, token_type_ids, target_mask, attention_mask, label, instance_id, candidates = self.data[index]
        return instance_id, input_id, candidates, attention_mask, target_mask, token_type_ids, label 
    
    def __len__(self):
        return len(self.data)
    


class CoarseGrainedDataset(Dataset):
    """
    Dataset for coarse-grained sense disambiguation
    
    Args:
    file_path (str): path to the json file containing the dataset
    """

    def __init__(self, file_path: str):
        self.data = self.load_data(file_path)
        
    def load_data(self, file_path: str):
        entries = []
        i = 0
        with open(file_path, 'r') as f:
            row = json.load(f)

        for idx, value in row.items():
            i += 1
            if config.DEBUG and i==100:
                break
            instance_ids = value["instance_ids"]
            words = value["words"]
            lemmas = value["lemmas"]
            pos_tags = value["pos_tags"]
            candidates = value["candidates"]
            senses = value["senses"]

            # set the sense labels for the metrics calculation
            set_sense_labels(idx,instance_ids, senses)

            # for each target word, for each candidate sense, generate a gloss pair
            for c_idx in candidates.keys():
                
                # use the fine candidates to generate the gloss pairs
                for idx in config.coarse_to_fine[candidates[c_idx]]:
                    pos_tag = pos_tags[int(idx)]
                    entries.extend(generate_glossBERT_pairs(words, candidates[idx], int(idx), senses[idx][0], pos_tag, instance_ids[idx], ws=config.WS))
        return entries
    
    def __getitem__(self, index: int):
        input_id, token_type_ids, target_mask, attention_mask, label, instance_id, candidates = self.data[index]
        return instance_id, input_id, candidates, attention_mask, target_mask, token_type_ids, label 
    
    def __len__(self):
        return len(self.data)


def load_map(file_path: str):
    """
    Load the mapping between coarse and fine senses
    """
    with open(file_path, 'r') as f:
        row = json.load(f)
    return row


def load_fine_definitions(file_path: str):
    """
    Load the fine definitions and the mapping between coarse and fine senses
    """
    mapping = load_map(file_path)
    fine_definitions = {}
    coarse_to_fine = {}
    for coarse, fine_list in mapping.items():
        coarse_to_fine[coarse] = []
        for fine_tuple in fine_list:
            (fine, definition) = list(fine_tuple.items())[0]
            fine_definitions[fine] = definition
            coarse_to_fine[coarse].append(fine)
    return fine_definitions, coarse_to_fine

def load_definitions(candidates: dict, mapping=None):
    """
    Load the definitions for the candidates

    Args:
    - candidates: a dictionary containing the candidates for each target word
    - mapping: a dictionary containing the mapping between the candidates and their definitions

    Returns:
    - definitions_map: a dictionary containing the definitions for each candidate

    """
    if mapping is None:
        mapping = load_map()
    definitions_map = {}
    for idx, cands in candidates.items():
        definitions_map[idx] = [mapping[candidate] for candidate in cands]

    return definitions_map

def set_sense_labels(idx, instance_ids, senses):
    """
    Set the sense labels for the metrics calculation

    Args:
    - idx: an integer representing the index of the sentence
    - instance_ids: a dictionary containing the instance ids for each target word
    - senses: a dictionary containing the sense labels for each target word
    """
    config.label_pairs_fine[idx] = {}
    for i, instance_id in instance_ids.items():
        config.label_pairs_fine[idx][instance_id] = config.fine_to_coarse[senses[i][0]]

