import sys
sys.path.append("./")
sys.path.append("./hw2/stud/")
import torch
from torch.utils.data import Dataset, DataLoader
import json
import config
import os
from utils import generate_glossBERT_pairs

class CoarseGrainedDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        
    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            row = json.load(f)
        return list(row.items())

    def __getitem__(self, index):
        key, value = self.data[index]
        
        words = value["words"]
        lemmas = value["lemmas"]
        pos_tags = value["pos_tags"]
        candidates = value["candidates"]
        senses = value["senses"]

        return words, lemmas, pos_tags, candidates, senses
    
    def __len__(self):
        return len(self.data)



def load_map(file_path):
    # print current working directory
    # print(os.getcwd())
    with open(file_path, 'r') as f:
        row = json.load(f)
    return row


def load_fine_definitions(file_path):
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

def load_definitions(candidates, mapping=None):
    if mapping is None:
        mapping = load_map()
    # from a dict of candidates {1: [cand11, cand12], 2: [cand21]} returns a dict of definitions
    definitions_map = {}
    for idx, cands in candidates.items():
        definitions_map[idx] = [mapping[candidate] for candidate in cands]

    return definitions_map


class FineGrainedDataset(Dataset):

    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        entries = []
        i = 0
        with open(file_path, 'r') as f:
            row = json.load(f)

        for idx, value in row.items():
            i += 1
            if i==10:
                break
            instance_ids = value["instance_ids"]
            words = value["words"]
            lemmas = value["lemmas"]
            pos_tags = value["pos_tags"]
            candidates = value["candidates"]
            senses = value["senses"]

            set_sense_labels(idx,instance_ids, senses)

            for idx in candidates.keys():
                pos_tag = pos_tags[int(idx)]

                entries.extend(generate_glossBERT_pairs(words, candidates[idx], int(idx), senses[idx][0], pos_tag, instance_ids[idx], ws=config.WS))

        return entries
    
    def __getitem__(self, index):
        input_id, token_type_ids, target_mask, attention_mask, label, instance_id, candidates = self.data[index]
        return instance_id, input_id, candidates, attention_mask, target_mask, token_type_ids, label 
    
    def __len__(self):
        return len(self.data)
    




def set_sense_labels(idx, instance_ids, senses):
    config.label_pairs_fine[idx] = {}
    for i, instance_id in instance_ids.items():
        config.label_pairs_fine[idx][instance_id] = config.fine_to_coarse[senses[i][0]]


# if __name__ == "__main__":
#     # Esempio di utilizzo
#     # dataset = CoarseGrainedDataset("./data/coarse-grained/train_coarse_grained.json")
#     dataset = FineGrainedDataset("./data/data/fine-grained/test_fine_grained.json")
#     mapping = load_map("./data/map/coarse_fine_defs_map.json")

#     definitions = load_fine_definitions()

#     print(len(dataset))
#     print(dataset[0])

