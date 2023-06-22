import sys
sys.path.append("./hw2/stud/")
import torch
import config
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
bert = AutoModel.from_pretrained('roberta-base')

def seed_everything(seed = 42):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    words_batch, lemmas, pos_tags, candidates_batch, senses_batch = zip(*batch)

    # Initialize lists to store the input sequences and labels
    input_sequences = []
    attention_masks = []
    all_candidates = []
    labels = []

    for i,words in enumerate(words_batch):
        candidates = candidates_batch[i]
        senses = senses_batch[i]
        for idx in candidates.keys():
            labels.append(config.sen_to_idx[senses[idx][0]])
            input_sequence = ['<s>'] + words[:int(idx)] + ['<d>', words[int(idx)], '</d>'] + words[int(idx)+1:]
            for candidate in candidates[idx]:
                input_sequence.extend(['<def>'] + config.definitions[candidate].split())
            for other_idx in candidates.keys():
                if other_idx != idx:
                    input_sequence.extend(['<GT>'] + config.definitions[senses[other_idx][0]].split())
            all_candidates.append([config.sen_to_idx[c] for c in candidates[idx]])
            input_sequence.append('</s>')
            input_sequences.append(input_sequence)
            attention_mask = [1 for _ in range(len(input_sequence))]
            attention_masks.append(attention_mask)


    max_len = max([len(attention_masks[i]) for i in range(len(attention_masks))])
    # Tokenize the input sequences
    input_ids = [tokenizer.encode(" ".join(input_sequence), truncation=True, padding='max_length', max_length=max_len) for input_sequence in input_sequences]
    # pad the attention masks with 0

    # Convert the input ids and labels to tensors
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = pad_sequence([torch.tensor(attention_mask) for attention_mask in attention_masks], batch_first=True, padding_value=0).to(config.DEVICE)
    candidates = pad_sequence([torch.tensor(candidate) for candidate in all_candidates], batch_first=True, padding_value=-1).to(config.DEVICE)
    return input_ids, attention_masks, candidates, labels



def compute_definition_embeddigns():
    batch_size = 100
    sense_embeddings = []

    # Loop over the senses in batches
    for i in range(0, len(config.sen_to_idx), batch_size):
        batch_senses = list(config.sen_to_idx.keys())[i:i+batch_size]
        batch_definitions = [config.definitions[sense] for sense in batch_senses]

        # Tokenize the definitions and convert to tensors
        inputs = tokenizer(batch_definitions, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # inputs = {name: tensor.to(config.DEVICE) for name, tensor in inputs.items()}

        # Compute the BERT embeddings
        with torch.no_grad():
            outputs = bert(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        sense_embeddings.extend(embeddings.cpu().numpy())

    sense_embeddings = np.array(sense_embeddings)

    return sense_embeddings

def sentence_embeddings(sen):
    input_ids = tokenizer.encode(sen, return_tensors='pt')
    outputs = bert(input_ids)
    return outputs.last_hidden_state.mean(dim=1)
