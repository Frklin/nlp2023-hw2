import sys
sys.path.append("./")
sys.path.append("./hw2/stud/")
from torch.utils.data import DataLoader
import config
import numpy as np
from utils import *
from load import *
from glossBERT import GlossBERT
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger



if __name__ == '__main__':

    seed_everything(config.SEED)

    train_dataset = FineGrainedDataset(config.FINE_TRAIN_DATA)
    val_dataset = FineGrainedDataset(config.FINE_VAL_DATA)
    test_dataset = FineGrainedDataset(config.FINE_TEST_DATA)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=glossBERT_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=glossBERT_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=glossBERT_collate_fn)

    mapping = load_map(config.MAP_PATH)

    model = GlossBERT()

    wandb_logger = WandbLogger(name='glossBERT', project='glossBERT')
    
    trainer = Trainer(gpus=1, max_epochs=config.EPOCHS, callbacks=[ModelCheckpoint(monitor='val_loss')], logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
