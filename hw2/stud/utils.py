import sys
sys.path.append("./hw2/stud/")
import torch
import config
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel, BertTokenizer

tokenizer = AutoTokenizer.from_pretrained('roberta-base') # SEP TOKEN AND CLS , cls_token=config.CLS_TOKEN
bert = AutoModel.from_pretrained('roberta-base')
#print special tokens
#print(tokenizer.all_special_tokens)

def seed_everything(seed = 42):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def glossBERT_collate_fn(batch):
    words_batch, lemmas, pos_tags, candidates_batch, labels, idx, instance_id, eos_idx = zip(*batch)

    sentence = words_batch[4]
    sentence = " ".join(sentence)
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    encodings = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    word_idx = encodings.word_ids()

    max_len = max([len(input_sequence) for input_sequence in words_batch])
    encodings = [tokenizer(" ".join(input_sequence), truncation=True, padding='max_length', max_length=max_len) for input_sequence in words_batch]
    input_ids = [enc["input_ids"] for enc in encodings]
    attention_mask = [enc["attention_mask"] for enc in encodings]
    word_idx = [enc.word_ids() for enc in encodings]
    # find the target_indexes 
    indeces = []
    eoses = []
    for i, sen in enumerate(word_idx):  
        target_indexes = []
        for j in range(idx[i],len(sen)):
            if sen[j] == idx[i]:
                target_indexes.append(j)
            if sen[j] == eos_idx[i]:
                eoses.append(j)
                break
        #create a tensor of 0 with 1 in the target_indexes
        t = torch.zeros(len(sen))
        t[target_indexes] = 1
        indeces.append(target_indexes)

    # create the token_type_ids
    token_type_ids = []
    for i,eos in enumerate(eos_idx):
        # create a tensor of 0 with 1 integers
        t = [0] * len(word_idx[0]) #torch.zeros(len(word_idx[0])).long()
        t[eos:] = [1] * (len(word_idx[0]) - eos)
        # put 0 where the attention mask is 0
        t = [0 if attention_mask[i][j] == 0 else t[j] for j in range(len(t))]
        token_type_ids.append(t)

    # pad everything
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_mask)
    # indeces = torch.tensor(indeces)
    labels = torch.tensor(labels)
    # indeces = pad_sequence(indeces, batch_first=True, padding_value=0).to(config.DEVICE)
    # word_idx = torch.tensor(word_idx, dtype=torch.long).to(config.DEVICE)
    # transform it to a tensor
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(config.DEVICE)

    return input_ids, labels, attention_masks, indeces, candidates_batch, instance_id, token_type_ids


def consec_collate_fn(batch):
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









# def glossBERT_collate_fn(batch):
#     words_batch, lemmas, pos_tags, candidates_batch, senses_batch = zip(*batch)

#     input_sequences = []
#     sense_ids = []
#     attention_masks = []
#     pos_tags = []
#     lemmas = []
#     labels = []

#     for i,words in enumerate(words_batch):
#         # sentences = []
#         candidates = candidates_batch[i]
#         senses = senses_batch[i]

#         for idx in candidates.keys():
#             query_word = words[int(idx)]
#             base_sentence = words[:int(idx)] + [config.DELIMITER_TOKEN + words[int(idx)] + config.DELIMITER_TOKEN] + words[int(idx)+1:] + [config.SEP_TOKEN] + [query_word + ":"]
#             for candidate in candidates[idx]:
#                 sense_ids.append(candidate)
#                 definition = config.definitions[candidate]
#                 # sentence = base_sentence +  " " + definition + " " + config.SEP_TOKEN
#                 sentence = base_sentence +  definition.split() + [config.SEP_TOKEN]

#                 labels.append(1 if candidate == senses[idx][0] else 0)

#                 input_sequences.append(sentence)
#                 # attention_mask = [1 for _ in range(len(sentence))]
#                 # attention_masks.append(attention_mask)

                
#     max_len = max([len(input_sequence) for input_sequence in input_sequences])
#     encodings = [tokenizer(" ".join(input_sequence), truncation=True, padding='max_length', max_length=max_len) for input_sequence in input_sequences]
#     input_ids = [enc["input_ids"] for enc in encodings]
#     attention_mask = [enc["attention_mask"] for enc in encodings]

#     # pad everything
#     input_ids = torch.tensor(input_ids)
#     sense_ids = torch.tensor([config.sen_to_idx[sense_id] for sense_id in sense_ids])
#     attention_masks = torch.tensor(attention_mask)

#     labels = torch.tensor(labels)

#     return input_ids, sense_ids, labels, attention_masks

