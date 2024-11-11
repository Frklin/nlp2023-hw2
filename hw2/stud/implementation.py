import sys
sys.path.append("../")
sys.path.append("hw2")
sys.path.append("hw2/stud")
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer
from glossBERT import GlossBERT
from utils import _get_ids
import torch
import torch.nn.functional as F

import config
from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel().to(device)


class RandomBaseline(Model):

    def __init__(self):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        pass

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        return [[np.random.choice(candidates) for candidates in sentence_data["candidates"].values()]
                for sentence_data in sentences]


class StudentModel(Model):

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, do_lower_case=True)
        
        self.model = GlossBERT.load_from_checkpoint(config.LOAD_MODEL_PATH)

        self.model.eval()
    
    def predict(self, sentences: List[Dict]) -> List[List[str]]:

        # generate all the gloss pairs
        gloss_pairs = self.get_gloss_pairs(sentences)

        # if no gloss pairs are found, return empty list
        if len(gloss_pairs) == 0:
            return []
        
        # encode the gloss pairs
        instance_ids, input_ids, candidates, attention_masks, token_type_ids, target_masks = self.encode_gloss_pairs(gloss_pairs)
        
        # get the predictions
        with torch.no_grad():
            model_predictions = self.model.predict(instance_ids, input_ids, candidates, attention_masks, token_type_ids, target_masks)

        # reconstruct the predictions to align with the input sentences
        predictions = self.reconstruct_predictions(model_predictions)

        return predictions

    def get_gloss_pairs(self, sentences: List[Dict]) -> List[List[str]]:
        """
        Generate gloss pairs from the input sentences

        Args:
        - sentences: a list of dictionaries, each containing the following keys:
            - instance_ids: a dictionary mapping indices to instance ids
            - words: a list of words
            - pos_tags: a list of part-of-speech tags
            - candidates: a dictionary mapping indices to lists of candidate senses

        Returns:
        - gloss_pairs: a list of tuples, each containing the following elements:
            - base_sentence: the input sentence with the weak supervision
            - gloss: a string representing the gloss
            - target_idx: an integer representing the index of the target word
            - instance_id: a string representing the instance id
            - candidate: a string representing the candidate sense
        """
        gloss_pairs = []

        for idx in range(len(sentences)):
            sentence_data = sentences[idx]
            instance_id = sentence_data["instance_ids"]
            words = sentence_data["words"]
            pos_tags = sentence_data["pos_tags"]
            candidates = sentence_data["candidates"]

            for idx in candidates.keys():
                
                # wrap the target word in "" for the weak supervision
                base_sentence = words[:int(idx)] + ["\""] + [words[int(idx)]] + ["\""] + words[int(idx)+1:]
                
                pos_tag = pos_tags[int(idx)] 

                for candidate in candidates[idx]:
                    
                    # if no mapping for the candidate is found, skip
                    if candidate not in config.coarse_to_fine:
                        continue

                    # from coarse cluster turn to fine cluster and iterate over each candidate
                    for fine_candidate in config.coarse_to_fine[candidate]:

                        candidate_pos = fine_candidate.split(".")[1]

                        # if the pos tag of the candidate is not the same as the pos tag of the target word, skip
                        if config.pos_map[candidate_pos] != pos_tag:
                            continue

                        # load the definition of the candidate and construct the gloss
                        definition = config.definitions[fine_candidate]
                        gloss = f"{words[int(idx)]} : {definition}"

                        gloss_pairs.append((base_sentence, gloss, int(idx), instance_id[idx], fine_candidate))                


        return gloss_pairs
    
    def encode_gloss_pairs(self, gloss_pairs: List[List[str]]) -> List[List[int]]:
        """
        Encode the gloss pairs using the model's tokenizer and pad the sequences

        Args:
        - gloss_pairs: a list of tuples, each containing the following elements:
            - base_sentence: the input sentence with the weak supervision
            - gloss: a string representing the gloss
            - target_idx: an integer representing the index of the target word
            - instance_id: a string representing the instance id
            - candidate: a string representing the candidate sense

        Returns:
        - input_ids: a tensor of input ids
        - token_type_ids: a tensor of token type ids
        - target_masks: a tensor of target masks
        - attention_masks: a tensor of attention masks
        - instance_ids: a list of instance ids
        - candidates: a list of candidate senses
        """

        input_ids = []
        token_type_ids = []
        target_masks = []
        attention_masks = []
        instance_ids = []
        candidates = []

        for sentence, gloss, target_idx, instance_id, candidate in gloss_pairs:
            # get the input ids, token type ids, target masks, and attention masks
            input_id, token_type_id, target_mask, attention_mask = _get_ids(sentence, gloss, target_idx)
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            target_masks.append(target_mask)
            attention_masks.append(attention_mask)
            instance_ids.append(instance_id)
            candidates.append(candidate)

        # find the maximum length of the input sequences for the padding
        max_len = max([len(input_sequence) for input_sequence in input_ids])

        # pad the sequences
        input_ids = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in input_ids])
        token_type_ids = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in token_type_ids])
        target_masks = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in target_masks])
        attention_masks = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in attention_masks])

        return instance_ids, input_ids, candidates, attention_masks, token_type_ids, target_masks

    def reconstruct_predictions(self, model_predictions: Dict[str, str]) -> List[List[str]]:
        """
        Reconstruct the predictions to align with the input sentences
        ["d014.s014.t000":select.v.01 , "d014.s014.t004":run.v.01 ...] -> [["select.v.01", "run.v.01"], ["..."]
        Args:
        - model_predictions: a dictionary mapping instance ids to predictions

        Returns:
        - predictions: a list of lists of predictions
        """
        predictions = []
        last_sentence_instance = None

        for instance_id, pred in model_predictions.items():
            # retrieve the sentence instance i.e. "d014.s014.t000" -> "d014.s014"
            sentence_instance = ".".join([instance_id.split(".")[0], instance_id.split(".")[1]])

            # if the sentence instance is different from the last one, append a new list to the predictions
            if sentence_instance != last_sentence_instance:
                predictions.append([])
                last_sentence_instance = sentence_instance

            predictions[-1].append(pred)

        return predictions
    
    def to(self, device: str):
        """
        Function to move the model to the device
        """
        self.model.to(device)
        return self
