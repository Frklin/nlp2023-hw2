import sys
sys.path.append("./")
sys.path.append("./hw2/stud/")
import torch
from torch.utils.data import Dataset, DataLoader
import json
import config
import os

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

class FineGrainedDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        
    def load_data(self, file_path):
        entries = []
        with open(file_path, 'r') as f:
            row = json.load(f)
            # create an entry for each candidate
            for key, value in row.items():
                base_words = value["words"]
                lemmas = value["lemmas"]
                pos_tags = value["pos_tags"]
                candidates = value["candidates"]
                senses = value["senses"]
                for idx in candidates.keys():
                    for candidate in candidates[idx]:
                        base_sentence = base_words[:int(idx)] + [config.DELIMITER_TOKEN + base_words[int(idx)] + config.DELIMITER_TOKEN] + base_words[int(idx)+1:] + [config.SEP_TOKEN] + [base_words[int(idx)] + ":"]
                        for candidate in candidates[idx]:
                            label = 1 if candidate == senses[idx][0] else 0
                            definition = config.definitions[candidate]
                            sentence = base_sentence +  definition.split() + [config.SEP_TOKEN]
                            entries.append((sentence, lemmas, pos_tags, candidates, senses, label))
        return entries

    def __getitem__(self, index):
        words, lemmas, pos_tags, candidates, senses, label = self.data[index]
        # words = value["words"]
        # lemmas = value["lemmas"]
        # pos_tags = value["pos_tags"]
        # candidates = value["candidates"]
        # senses = value["senses"]

        return words, lemmas, pos_tags, candidates, senses, label
    
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
    for coarse, fine_list in mapping.items():
        for fine_tuple in fine_list:
            (fine, definition) = list(fine_tuple.items())[0]
            fine_definitions[fine] = definition
    return fine_definitions

def load_definitions(candidates, mapping=None):
    if mapping is None:
        mapping = load_map()
    # from a dict of candidates {1: [cand11, cand12], 2: [cand21]} returns a dict of definitions
    definitions_map = {}
    for idx, cands in candidates.items():
        definitions_map[idx] = [mapping[candidate] for candidate in cands]

    return definitions_map

# if __name__ == "__main__":
#     # Esempio di utilizzo
#     dataset = CoarseGrainedDataset("./data/coarse-grained/train_coarse_grained.json")
#     # dataset = FineGrainedDataset("./data/data/fine-grained/test_fine_grained.json")
#     mapping = load_map("./data/map/coarse_fine_defs_map.json")

#     definitions = load_fine_definitions()



