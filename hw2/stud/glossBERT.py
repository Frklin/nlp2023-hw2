

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel, RobertaForTokenClassification, BertForTokenClassification
import config


import torch
import torch.nn as nn
import pytorch_lightning as pl

class GlossBERT(pl.LightningModule):
    def __init__(self, pretrained_model_name='roberta-base'):
        super(GlossBERT, self).__init__()
        
        # Initialize the BERT model and tokenizer
        self.roberta = RobertaForTokenClassification.from_pretrained(pretrained_model_name, num_labels=2)
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.accuracy_train = []
        self.accuracy_val = []

        self.train_predictions = {}
        self.val_predictions = {}

        # Define the classifier
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2) #config.num_classes_fine)  # num_classes should be defined

        #loss
        self.loss = nn.CrossEntropyLoss() #ignore_index=0
        
    def forward(self, input_ids, attention_mask, indices, token_type_ids):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        #cls output
        cls_output = outputs.last_hidden_state[:, 0, :]

        # do a mean of the embeddings
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

        # take the embeddings of the ambiguous word
        target_embeddings = []

        # Loop over the batch
        for i, sentence_indices in enumerate(indices):
            # Extract the embeddings corresponding to the target words in the sentence
            embeddings = outputs.last_hidden_state[i, sentence_indices, :]
            
            # Average the embeddings along dimension 0 (since sentence_indices is a list of indices)
            target_embedding = torch.mean(embeddings, dim=0)
            
            # Append the averaged embedding to the list
            target_embeddings.append(target_embedding)

        # Convert the list of averaged embeddings to a tensor
        target_embeddings = torch.stack(target_embeddings)

        # Prediction
        logits = self.classifier(target_embeddings)  # cls_output
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, indices, candidates, instance_id, token_type_ids = batch

        # for each input ids find the first occurence of the delimiter token


        # logits = self.roberta(input_ids, attention_mask, labels=labels, target_mask=indices)
        logits = self.forward(input_ids, attention_mask, indices, token_type_ids)
        loss = self.loss(logits, labels)   
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy,  prog_bar=True)

        self.accuracy_train.append(accuracy)

        for i, id in enumerate(instance_id):
            if id not in self.train_predictions.keys():
                self.train_predictions[id] = []
            self.train_predictions[id].append((candidates[i], float(logits[i, 1])))
        # self.write_prediction(instance_id, candidates, logits[:, 1])

        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids, labels, attention_mask, indices, candidates, instance_id, token_type_ids = batch
            
            logits = self.forward(input_ids, attention_mask, indices, token_type_ids)
            # logits = self.roberta(input_ids, attention_mask, labels=None, target_mask=indices)
            loss = self.loss(logits, labels)
            accuracy = (logits.argmax(dim=-1) == labels).float().mean()
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', accuracy, prog_bar=True)
            self.accuracy_val.append(accuracy)

            # if instance_id is present in the predictions, append the new predictions
            # else create a new entry
            for i, id in enumerate(instance_id):
                if id not in self.val_predictions.keys():
                    self.val_predictions[id] = []
                self.val_predictions[id].append((candidates[i], float(logits[i, 1])))
                # self.write_prediction(id, candidates[i], logits[i, 1])
            # self.write_prediction(instance_id, candidates, logits[:, 1])

        return {'loss': loss}
    
    def predict(self, sentence, attention_mask, indices):
        # take in input the tokens and the glosses

        # create the different possibilities

        # get the logits for each possibility

        # return the best one
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer
    
    def training_epoch_end(self, outputs) -> None:
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        accuracy_train_mean = sum(self.accuracy_train) / len(self.accuracy_train)
        accuracy_val_mean = sum(self.accuracy_val) / len(self.accuracy_val)
        self.accuracy_train = []
        self.accuracy_val = []

        print("Epoch: {}, Train loss: {}, Train accuracy: {}, Val accuracy: {}".format(self.current_epoch, loss, accuracy_train_mean, accuracy_val_mean))
        # train metrics
        for instance_id, pred in self.train_predictions.items():
                pred.sort(key=lambda x: x[1], reverse=True)
                self.train_predictions[instance_id] = pred[0][0]
        
        for instance_id, pred in self.val_predictions.items():
                pred.sort(key=lambda x: x[1], reverse=True)
                self.val_predictions[instance_id] = pred[0][0]

        self.print_metrics()

        self.train_predictions = {}
        self.val_predictions = {}
            
            

    def write_prediction(self, istance_id, prediction, value):
        with open(config.PREDICTION_PATH, 'a') as f:
            for i in range(len(prediction)):
                f.write(str(istance_id[i]) + ',' + str(prediction[i]) + ',' + str(value[i].item()) + '\n')

    def print_metrics(self):
        correct = 0
        total = 0
        for instance_id, pred in self.train_predictions.items():
            if pred == config.label_pairs_fine[instance_id]:
                correct += 1
            total += 1
        train_accuracy = round(correct/total, 3)
        print("Accuracy for train: {}".format(train_accuracy))

        correct = 0
        total = 0
        for instance_id, pred in self.val_predictions.items():
            if pred == config.label_pairs_fine[instance_id]:
                correct += 1
            total += 1
        val_accuracy = round(correct/total, 3)
        print("Accuracy for validation: {}".format(val_accuracy))


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




