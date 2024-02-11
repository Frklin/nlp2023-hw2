

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel, RobertaForSequenceClassification, BertForTokenClassification, AutoConfig
import config

import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl

class GlossBERT(pl.LightningModule):
    def __init__(self, pretrained_model_name=config.MODEL_NAME):
        super(GlossBERT, self).__init__()
        
        # Initialize the BERT model and tokenizer
        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)  # config

        # for param in self.bert.parameters():
        #     param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.train_losses = []
        self.accuracy_train = []
        self.train_predictions = {}
        self.final_train_predictions = {}

        self.val_losses = []
        self.val_predictions = {}
        self.final_val_predictions = {}
        self.accuracy_val = []


        # Define the classifier
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2) #config.num_classes_fine)  # num_classes should be defined
        self.dropout = nn.Dropout(0.1)

        #loss
        self.loss = nn.CrossEntropyLoss() #ignore_index=0
        # self.loss = nn.BCELoss()
        
    def forward(self, input_ids, attention_mask, token_type_ids, target=None):

        out = self.bert(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)
        
        target_embeddings = []
        embeddings = out.last_hidden_state        

        if config.MODE == "TARGET":

            target_embeddings = embeddings[target]   


        elif config.MODE == "CLS":
            target_embeddings = embeddings[:, 0, :]
        
        else:
            raise ValueError("MODE not valid")
        
        # target_embeddings = self.dropout(target_embeddings)

        logits = self.classifier(target_embeddings)

        return logits

    def training_step(self, batch, batch_idx):
        instance_ids, input_ids, candidates, attention_masks, target_masks, token_type_ids, labels = batch
            
        logits = self.forward(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_masks,
                        target=target_masks)
        
        loss = self.loss(logits, labels)

        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc_batch', accuracy,  prog_bar=True)

        self.accuracy_train.append(accuracy)
        self.train_losses.append(loss)

        for i, sentence_id in enumerate(instance_ids):
            if sentence_id not in self.train_predictions.keys():
                self.train_predictions[sentence_id] = []
            self.train_predictions[sentence_id].append((candidates[i], float(logits[i, 1])))
            # self.write_prediction(sentence_id, candidates[i], logits[i, 1], labels[i] , "TRAIN")

        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):

        instance_ids, input_ids, candidates, attention_masks, target_masks, token_type_ids, labels = batch

        logits = self.forward(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                    target=target_masks)
        
        loss = self.loss(logits, labels)

        accuracy = (logits.argmax(dim=-1) == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc_batch', accuracy, prog_bar=True)
        self.accuracy_val.append(accuracy)
        self.val_losses.append(loss)

        # LOAD THE RESULTS 
        for i, sentence_id in enumerate(instance_ids):
            if sentence_id not in self.val_predictions.keys():
                self.val_predictions[sentence_id] = []
            self.val_predictions[sentence_id].append((candidates[i], float(logits[i, 1])))
            # self.write_prediction(sentence_id, candidates[i], logits[i, 1], labels[i] ,"VAL")

        return {'loss': loss}
    
    def predict(self, input_data):

        instance_ids, input_ids, candidates, attention_masks, token_type_ids = input_data

        logits = self.forward(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks)
        
        all_predictions = {}
        predictions = {}


        for i, sentence_id in enumerate(instance_ids):
            if sentence_id not in all_predictions.keys():
                all_predictions[sentence_id] = []
            all_predictions[sentence_id].append((candidates[i], float(logits[i, 1])))

        for instance_id, pred in all_predictions.items():
            pred.sort(key=lambda x: x[1], reverse=True)
            predictions[instance_id] = config.fine_to_coarse[pred[0][0]]
        
        return predictions


    
    def on_train_epoch_end(self):
        loss = sum(self.train_losses) / len(self.train_losses)
        accuracy_train_mean = sum(self.accuracy_train) / len(self.accuracy_train)


        print("Epoch: {}, Train loss: {}, Train accuracy: {}".format(self.current_epoch, loss, accuracy_train_mean))

        wandb.log({'train_epoch_loss': loss, 'train_accuracy': accuracy_train_mean})

        for instance_id, pred in self.train_predictions.items():
                pred.sort(key=lambda x: x[1], reverse=True)
                self.final_train_predictions[instance_id] = config.fine_to_coarse[pred[0][0]]

        self.print_train_metrics()

        self.train_losses = []
        self.accuracy_train = []
        self.train_predictions = {}
        self.final_train_predictions = {}
            
    def on_validation_epoch_end(self):
        loss = sum(self.val_losses) / len(self.val_losses)
        accuracy_val_mean = sum(self.accuracy_val) / len(self.accuracy_val)

        print("Epoch: {}, Validation loss: {}, Validation accuracy: {}".format(self.current_epoch, loss, accuracy_val_mean))

        wandb.log({'val_epoch_loss': loss, 'val_accuracy': accuracy_val_mean})

        for instance_id, pred in self.val_predictions.items():
                pred.sort(key=lambda x: x[1], reverse=True)
                self.final_val_predictions[instance_id] = config.fine_to_coarse[pred[0][0]]

        self.print_val_metrics()

        self.val_predictions = {}
        self.val_losses = []
        self.accuracy_val = []
        self.final_val_predictions = {}



    def write_prediction(self, instance_id, prediction, value, label, step):
        with open(config.PREDICTION_PATH, 'a') as f:
                f.write('(' +step+') '+str(instance_id) + ', ' + str(prediction) + ',  ' + str(value.item())[:5] + f", ({label})\n")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        return optimizer
    

    def print_train_metrics(self):
        correct = 0
        total = 0
        for instance_id, pred in self.final_train_predictions.items():
            if pred == config.single_label_pairs_fine[instance_id]:
                correct += 1
            total += 1
        train_accuracy = round(correct/total, 3) if total > 0 else 0
        # print("Accuracy for train: {}".format(train_accuracy))
        wandb.log({'train_accuracy': train_accuracy})

    def print_val_metrics(self):
        correct = 0
        total = 0
        for instance_id, pred in self.final_val_predictions.items():
            if pred == config.single_label_pairs_fine[instance_id]:
                correct += 1
            total += 1
        val_accuracy = round(correct/total, 3) if total > 0 else 0
        # print("Accuracy for validation: {}".format(val_accuracy))
        wandb.log({'val_accuracy': val_accuracy})
