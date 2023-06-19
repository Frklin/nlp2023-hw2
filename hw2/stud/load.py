import sys
sys.path.append("./")
sys.path.append("./hw2/stud/")
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import config

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
        self.data = []
        self.file_path = file_path
        self.load_data(file_path)
        
    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            row = json.load(f)
        self.data = row

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
    with open(file_path, 'r') as f:
        row = json.load(f)
    return row

def collate_fn(batch):
    words, lemmas, pos_tags, candidates, senses = zip(*batch)

    # compute the max len 
    # max_lenght = max([len(w) for w in words])
    words = torch.tensor(words)
    lemmas = [torch.tensor(l) for l in lemmas]
    pos_tags = [torch.tensor(p) for p in pos_tags]

    padded_words = pad_sequence(words, batch_first=True, padding_value=config.PAD_TOKEN).to(config.DEVICE)
    padded_lemmas = pad_sequence(lemmas, batch_first=True, padding_value=config.PAD_TOKEN).to(config.DEVICE)
    padded_pos_tags = pad_sequence(pos_tags, batch_first=True, padding_value=config.PAD_TOKEN).to(config.DEVICE)

    return padded_words, padded_lemmas, padded_pos_tags, candidates, senses

def load_definitions(candidates, mapping):
    # from a dict of candidates {1: [cand11, cand12], 2: [cand21]} returns a dict of definitions
    definitions_map = {}
    for idx, cands in candidates.items():
        definitions_map[idx] = [mapping[candidate] for candidate in cands]

    return definitions_map

if __name__ == "__main__":
    # Esempio di utilizzo
    dataset = CoarseGrainedDataset("./data/data/coarse-grained/train_coarse_grained.json")
    # dataset = FineGrainedDataset("./data/data/fine-grained/test_fine_grained.json")
    
    mapping = load_map("./data/data/map/coarse_fine_defs_map.json")



