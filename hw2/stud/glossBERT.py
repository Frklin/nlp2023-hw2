

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import config


import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer

class GlossBERT(pl.LightningModule):
    def __init__(self, pretrained_model_name='roberta-base'):
        super(GlossBERT, self).__init__()
        
        # Initialize the BERT model and tokenizer
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        # Define the classifier
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes_fine)  # num_classes should be defined
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Prediction
        logits = self.classifier(cls_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, labels, attention_mask = batch
        # divide the values of input_ids by 10 unless they are 0

        logits = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    
    def validation_step(self, batch, batch_idx):
        # Implement your validation logic here
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer



# class ConSecModel(LightningModule):
#     def __init__(self, num_senses):
#         super(ConSecModel, self).__init__()

#         # Load the pre-trained model
#         self.bert = AutoModel.from_pretrained('roberta-base')

#         # Initialize the sense embeddings
#         self.sense_embeddings = nn.Embedding(num_senses, self.bert.config.hidden_size)

#         # Define a linear layer for the sense prediction
#         self.fc = nn.Linear(self.bert.config.hidden_size, num_senses)

#         # Define the loss function
#         self.loss_fn = nn.CrossEntropyLoss()

#     def forward(self, input_ids, attention_mask, candidate_indices):
#         # Generate contextualized word representations
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         word_representations = outputs.last_hidden_state

#         # Select the word representation of the ambiguous word
#         ambiguous_word_representations = word_representations[:, 0, :]

#         # Select the sense representations of the candidate senses
#         candidate_representations = self.sense_embeddings(candidate_indices)

#         # Compute the sense scores
#         sense_scores = torch.bmm(candidate_representations, ambiguous_word_representations.unsqueeze(-1)).squeeze(-1)

#         # Compute the sense probabilities
#         sense_probs = nn.functional.softmax(sense_scores, dim=-1)

#         return sense_probs
    

#     def training_step(self, batch, batch_idx):
#         input_ids, attention_mask, candidates, labels = batch
#         sense_representations, sense_probs = self(input_ids, attention_mask, candidates)
#         print(sense_probs)
#         loss = self.loss_fn(sense_probs.view(-1, sense_probs.size(-1)), labels.view(-1))
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         input_ids, attention_mask, candidates, labels = batch
#         sense_representations, sense_probs = self(input_ids, attention_mask, candidates)
#         loss = self.loss_fn(sense_probs.view(-1, sense_probs.size(-1)), labels.view(-1))
#         self.log('val_loss', loss)
    
#     def configure_optimizers(self):
#         # Define your optimizer
#         return torch.optim.Adam(self.parameters())

# class ESCModel(LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.linear = nn.Linear(self.bert.config.hidden_size, 1)
#         self.loss = nn.BCEWithLogitsLoss()

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         logits = self.linear(outputs.last_hidden_state)
#         return logits

#     def training_step(self, batch, batch_idx):
#         input_ids, attention_mask, labels = batch
#         logits = self(input_ids, attention_mask)
#         loss = self.loss(logits.view(-1), labels.view(-1))
#         self.log('train_loss', loss)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-5)




# def sense_vocabolary(definitions):

#     # Load the pre-trained model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#     model = AutoModel.from_pretrained('roberta-base')

#     # Initialize a dictionary to store the sense embeddings
#     sense_embeddings = {}

#     # For each cluster in your dataset
#     for cluster, senses in definitions.items():
#         # Initialize a list to store the embeddings for this cluster
#         cluster_embeddings = []

#         # For each sense in the cluster
#         for sense in senses:
#             # Get the definition of the sense
#             definition = list(sense.values())[0]

#             # Tokenize the definition
#             inputs = tokenizer(definition, return_tensors='pt')

#             # Generate an embedding for the definition
#             outputs = model(**inputs)
#             embedding = outputs.last_hidden_state.mean(dim=1)

#             # Add the embedding to the list
#             cluster_embeddings.append(embedding)

#         # Average the embeddings to get a single embedding for the cluster
#         cluster_embedding = torch.stack(cluster_embeddings).mean(dim=0)

#         # Store the cluster embedding in the dictionary
#         sense_embeddings[cluster] = cluster_embedding




