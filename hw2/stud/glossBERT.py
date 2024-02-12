import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import  AutoTokenizer, AutoModel
from metrics import compute_metrics
import wandb



class GlossBERT(pl.LightningModule):
    """
    GlossBERT model for word sense disambiguation. 
    GlossBERT enhances the traditional BERT model by incorporating gloss information directly into the training process. 
    This is achieved by fine-tuning a pre-trained BERT model on data using sentence-gloss pairs, thus enabling the model
    to learn the distinctions of word senses in varied contexts.
    """

    def __init__(self, pretrained_model_name: str = config.MODEL_NAME):
        """
        Initialization of the GlossBERT model.
        Args:
            pretrained_model_name (str): Name of the pre-trained model to use (bert-base-cased).
        """
        super(GlossBERT, self).__init__()
        
        # BERT model
        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)  
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name) # For fast debugging

        # Classifier
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES) 
        self.dropout = nn.Dropout(config.DROPOUT)

        # Loss function
        self.loss = nn.CrossEntropyLoss() 

        # Train Metrics
        self.train_losses: list = []
        self.accuracy_train: list = []
        self.train_predictions: set = {}      # {instance_id: [(candidate, confidence)]}
        self.final_train_predictions = {}     # {instance_id: prediction}

        # Validation Metrics
        self.val_losses: list = []
        self.accuracy_val: list = []
        self.val_predictions: set = {}        # {instance_id: [(candidate, confidence)]}
        self.final_val_predictions: set = {}  # {instance_id: prediction}
        
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            input_ids (torch.Tensor): Input token ids.                  [B, max_len]
            token_type_ids (torch.Tensor): Segment token ids.           [B, max_len]
            attention_mask (torch.Tensor): Attention mask.              [B, max_len]
            target (torch.Tensor) : Target mask.                        [B, max_len]
        Returns:
            torch.Tensor [B,2]: Output logits of glossBERT.
        """

        # BERT forward pass
        out = self.bert(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)
        
        target_embeddings = []
        embeddings = out.last_hidden_state       # [B, max_len, hidden_size]

        # Select the target word embeddings
        if config.MODE == "TARGET":
            target_embeddings = embeddings[target]   # [B, hidden_size]

        # Select the CLS token embeddings
        elif config.MODE == "CLS":
            target_embeddings = embeddings[:, 0, :]  # [B, hidden_size]
        
        else:
            raise ValueError("MODE not valid")
        
        # Apply dropout
        target_embeddings = self.dropout(target_embeddings)

        # Apply classifier
        logits = self.classifier(target_embeddings)

        return logits

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """
        Training step of the model.
        Args:
            batch (torch.Tensor): Input batch. [B, max_len]
            batch_idx (int): Batch index.
        Returns:
            dict: Dictionary with the loss.
        """

        instance_ids, input_ids, candidates, attention_masks, target_masks, token_type_ids, labels = batch
            
        # Forward pass
        logits = self.forward(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_masks,
                        target=target_masks)
        
        # Compute loss
        loss = self.loss(logits, labels)

        # Compute accuracy
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc_batch', accuracy,  prog_bar=True)
        self.accuracy_train.append(accuracy)
        self.train_losses.append(loss)

        # Load the results for the predictions in the training set
        for i, sentence_id in enumerate(instance_ids):
            if sentence_id not in self.train_predictions.keys():
                self.train_predictions[sentence_id] = []
            self.train_predictions[sentence_id].append((candidates[i], float(logits[i, 1])))
            
            if config.DEBUG:
                self.write_prediction(sentence_id, candidates[i], logits[i, 1], labels[i] , "TRAIN")

        return {'loss': loss}
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """
        Validation step of the model.
        Args:
            batch (torch.Tensor): Input batch. [B, max_len]
            batch_idx (int): Batch index.
        Returns:
            dict: Dictionary with the loss.
        """

        instance_ids, input_ids, candidates, attention_masks, target_masks, token_type_ids, labels = batch

        # Forward pass
        logits = self.forward(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                    target=target_masks)
        
        # Compute loss
        loss = self.loss(logits, labels)

        # Compute accuracy
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc_batch', accuracy, prog_bar=True)
        self.accuracy_val.append(accuracy)
        self.val_losses.append(loss)

        # Load the results for the predictions in the validation set 
        for i, sentence_id in enumerate(instance_ids):
            if sentence_id not in self.val_predictions.keys():
                self.val_predictions[sentence_id] = []
            self.val_predictions[sentence_id].append((candidates[i], float(logits[i, 1])))

            if config.DEBUG:
                self.write_prediction(sentence_id, candidates[i], logits[i, 1], labels[i], "VAL")

        return {'loss': loss}
    
    def on_train_epoch_end(self):
        """
        Method called at the end of the training epoch.

        It collects the predictions for the training set and computes the metrics.

        The metrics are logged to wandb.
        """

        # Compute the mean loss and accuracy
        loss = sum(self.train_losses) / len(self.train_losses)
        accuracy_train_mean = sum(self.accuracy_train) / len(self.accuracy_train)
        wandb.log({'train_epoch_loss': loss, 'train_accuracy': accuracy_train_mean})

        # Take the prediction with the highest confidence
        for instance_id, pred in self.train_predictions.items():
                pred.sort(key=lambda x: x[1], reverse=True)
                self.final_train_predictions[instance_id] = config.fine_to_coarse[pred[0][0]]

        # Compute the metrics
        compute_metrics(self.final_train_predictions, "train")

        # Reset the metrics
        self.train_losses = []
        self.accuracy_train = []
        self.train_predictions = {}
        self.final_train_predictions = {}
            
    def on_validation_epoch_end(self):
        """
        Method called at the end of the validation epoch.

        It collects the predictions for the validation set and computes the metrics.

        The metrics are logged to wandb.
        """

        # Compute the mean loss and accuracy
        loss = sum(self.val_losses) / len(self.val_losses)
        accuracy_val_mean = sum(self.accuracy_val) / len(self.accuracy_val)
        wandb.log({'val_epoch_loss': loss, 'val_accuracy': accuracy_val_mean})

        # Take the prediction with the highest confidence
        for instance_id, pred in self.val_predictions.items():
                pred.sort(key=lambda x: x[1], reverse=True)
                self.final_val_predictions[instance_id] = config.fine_to_coarse[pred[0][0]]

        # Compute the metrics
        compute_metrics(self.final_val_predictions, "val")

        # Reset the metrics
        self.val_predictions = {}
        self.val_losses = []
        self.accuracy_val = []
        self.final_val_predictions = {}

    def predict(self, instance_ids: list,
                input_ids: torch.Tensor,
                candidates: list,
                attention_masks: torch.Tensor,
                token_type_ids: torch.Tensor,
                target_mask: torch.Tensor) -> dict:
        """
        Predict the word sense for a given set of instances with a pretrained glossBERT model.
        Args:
            instance_ids (list): List of instance ids.
            input_ids (torch.Tensor): Input token ids.                  [B, max_len]
            candidates (list): List of candidate words.
            attention_masks (torch.Tensor): Attention mask.             [B, max_len]
            token_type_ids (torch.Tensor): Segment token ids.           [B, max_len]
            target_mask (torch.Tensor) : Target mask.                   [B, max_len]
        Returns:
            dict: Dictionary with the predictions.
        """

        # Forward pass
        logits = self.forward(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                    target=target_mask)
        
        all_predictions = {}
        predictions = {}

        # Load the results and group them by instance_id
        for i, sentence_id in enumerate(instance_ids):
            if sentence_id not in all_predictions.keys():
                all_predictions[sentence_id] = []
            all_predictions[sentence_id].append((candidates[i], float(logits[i, 1])))

        # Take the prediction with the highest confidence
        for instance_id, pred in all_predictions.items():
            pred.sort(key=lambda x: x[1], reverse=True)
            predictions[instance_id] = config.fine_to_coarse[pred[0][0]]
        
        return predictions

    def write_prediction(self, instance_id: int, prediction: str, value: float, label: int, step: str):
        with open(config.PREDICTION_PATH, 'a') as f:
                f.write('(' +step+') '+str(instance_id) + ', ' + str(prediction) + ',  ' + str(value.item())[:5] + f", ({label})\n")

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        Returns:
            torch.optim.AdamW: AdamW optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        return optimizer
    
