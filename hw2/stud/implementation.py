import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer
from glossBERT import GlossBERT
from load import set_sense_labels
from utils import generate_glossBERT_pairs, _get_ids
import torch
import torch.nn.functional as F

import config
from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel()


class RandomBaseline(Model):

    def __init__(self):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        pass

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        return [[np.random.choice(candidates) for candidates in sentence_data["candidates"].values()]
                for sentence_data in sentences]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, do_lower_case=True)
        
        self.model = GlossBERT()

    def predict(self, sentences: List[Dict]) -> List[List[str]]:

        gloss_pairs = self.get_gloss_pairs(sentences)

        instance_ids, input_ids, candidates, attention_masks, token_type_ids, target_masks = self.encode_gloss_pairs(gloss_pairs)

        model_predictions = self.model.predict(instance_ids, input_ids, candidates, attention_masks, token_type_ids, target_masks)

        # predictions = reconstruct_predictions(model_predictions, labels)

        



    def get_gloss_pairs(self, sentences: List[Dict]) -> List[List[str]]:
        gloss_pairs = []

        for idx in range(len(sentences)):
            sentence_data = sentences[idx]
            instance_id = sentence_data["instance_ids"]
            words = sentence_data["words"]
            pos_tags = sentence_data["pos_tags"]
            candidates = sentence_data["candidates"]

            for idx in candidates.keys():
                
                base_sentence = words[:int(idx)] + ["\""] + [words[int(idx)]] + ["\""] + words[int(idx)+1:]
                
                pos_tags = pos_tags[int(idx)] 

                for candidate in candidates[idx]:

                        candidate_pos = candidate.split(".")[1]

                        if config.pos_map[candidate_pos] != pos_tags:
                            continue

                        definition = config.definitions[candidate]
                        gloss = f"{words[int(idx)]} : {definition}"

                        gloss_pairs.append(base_sentence, gloss, int(idx), instance_id, candidate)                


        return gloss_pairs
    
    def encode_gloss_pairs(self, gloss_pairs: List[List[str]]) -> List[List[int]]:
        input_ids = []
        token_type_ids = []
        target_masks = []
        attention_masks = []
        instance_ids = []
        candidates = []

        for sentence, gloss, target_idx, instance_id, candidate in gloss_pairs:
            input_id, token_type_id, target_mask, attention_mask = _get_ids(sentence, gloss, target_idx)
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            target_masks.append(target_mask)
            attention_masks.append(attention_mask)
            instance_ids.append(instance_id)
            candidates.append(candidate)

        max_len = max([len(input_sequence) for input_sequence in input_ids])

        input_ids = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in input_ids])
        token_type_ids = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in token_type_ids])
        target_masks = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in target_masks])
        attention_masks = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in attention_masks])




        return instance_ids, input_ids, candidates, attention_mask, token_type_ids, target_masks
