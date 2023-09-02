import sys
sys.path.append("./")
sys.path.append("./hw2/stud/")
import torch
from torch.utils.data import Dataset, DataLoader
import json
import config
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('roberta-base') # SEP TOKEN AND CLS , cls_token=config.CLS_TOKEN
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
            i = 0
            # create an entry for each candidate
            for _, value in row.items():
                i += 1
                if i==10:
                    break
                instance_ids = value["instance_ids"]
                base_words = value["words"]
                lemmas = value["lemmas"]
                pos_tags = value["pos_tags"]
                candidates = value["candidates"]
                senses = value["senses"]
                for idx in candidates.keys():
                    pos = pos_tags[int(idx)]
                    base_sentence = [config.CLS_TOKEN] + base_words[:int(idx)] + [base_words[int(idx)]] + base_words[int(idx)+1:] + [config.SEP_TOKEN] + [base_words[int(idx)] + ":"]
                    # for candidate in candidates[idx]:
                    # sentence_tokens = tokenizer(" ".join(base_sentence))
                    for candidate in candidates[idx]:
                        #check the pos
                        candidate_pos = candidate.split(".")[1]
                        if config.pos_map[candidate_pos] != pos:
                            print(pos, candidate_pos)
                            continue
                        label = 1 if candidate == senses[idx][0] else 0
                        definition = config.definitions[candidate]
                        # gloss_tokens = tokenizer(definition)
                        config.label_pairs_fine[instance_ids[idx]] = candidate
                        # config.predictions[instance_ids[idx]] = []
                        # tokens = [config.CLS_TOKEN] + sentence_tokens["input_ids"] + [config.SEP_TOKEN]
                        # segments = [0] * len(tokens)
                        
                        # tokens += gloss_tokens["input_ids"] + [config.SEP_TOKEN]
                        # segments += [1] * (len(gloss_tokens["input_ids"]) + 1)

                        # print(tokens)
                        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
                        sentence = base_sentence +  definition.split() + [config.SEP_TOKEN]
                        entries.append((sentence, lemmas, pos_tags, candidate, label, int(idx)+1, instance_ids[idx], len(base_sentence)-1))
        return entries

    def __getitem__(self, index):
        words, lemmas, pos_tags, candidate, label, idx, instance_id, eos_idx = self.data[index]
        return words, lemmas, pos_tags, candidate, label, idx, instance_id, eos_idx
    
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
    coarse_to_grain = {}
    for coarse, fine_list in mapping.items():
        for fine_tuple in fine_list:
            (fine, definition) = list(fine_tuple.items())[0]
            fine_definitions[fine] = definition
            coarse_to_grain[coarse] = fine
    return fine_definitions, coarse_to_grain

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



