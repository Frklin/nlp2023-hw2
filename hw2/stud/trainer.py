# # import pytorch lightning
# import torch
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader

# from stud.load import WSDataset


# class WSDataModule(pl.LightningDataModule):
#     def __init__(self, batch_size, num_workers, train_file, val_file, test_file):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.train_file = train_file
#         self.val_file = val_file
#         self.test_file = test_file

#     def setup(self, stage=None):
#         self.train_dataset = WSDataset(self.train_file)
#         self.val_dataset = WSDataset(self.val_file)
#         self.test_dataset = WSDataset(self.test_file)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
#     def get_vocab_size(self):
#         return self.train_dataset.get_vocab_size()
    
#     def get_num_senses(self):
#         return self.train_dataset.get_num_senses()
    
#     def get_num_pos_tags(self):
#         return self.train_dataset.get_num_pos_tags()
    
#     def get_num_candidates(self):
#         return self.train_dataset.get_num_candidates()
    
#     def get_num_instances(self):
#         return self.train_dataset.get_num_instances()
    
#     def get_num_words(self):
#         return self.train_dataset.get_num_words()
    
#     def get_num_lemmas(self):
#         return self.train_dataset.get_num_lemmas()
    

# class WSDataModuleFineTune(pl.LightningDataModule):
#     def __init__(self, batch_size, num_workers, train_file, val_file, test_file):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.train_file = train_file
#         self.val_file = val_file
#         self.test_file = test_file

#     def setup(self, stage=None):
#         self.train_dataset = WSDataset(self.train_file)
#         self.val_dataset = WSDataset(self.val_file)
#         self.test_dataset = WSDataset(self.test_file)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
#     def get_vocab_size(self):
#         return self.train_dataset.get_vocab_size()
    
#     def get_num_senses(self):
#         return self.train_dataset.get_num_senses()
    
#     def get_num_pos_tags(self):
#         return self.train_dataset.get_num_pos_tags()
    
#     def get_num_candidates(self):
#         return self.train_dataset.get_num_candidates()
    
#     def get_num_instances(self):
#         return self.train_dataset.get_num_instances()
    
#     def get_num_words(self):
#         return self.train_dataset.get_num_words()
    
#     def get_num_lemmas(self):
#         return self.train_dataset.get_num_lemmas()
    
#     def get_num_fine_senses(self):
#         return self.train_dataset.get_num_fine_senses()
    
#     def get_num_fine_pos_tags(self):
#         return self.train_dataset.get_num_fine_pos_tags()
    
#     def get_num_fine_candidates(self):
#         return self.train_dataset.get_num_fine_candidates()
    
#     def get_num_fine_instances(self):
#         return self.train_dataset.get_num_fine_instances()
    
#     def get_num_fine_words(self):
#         return self.train_dataset.get_num_fine_words()
    
#     def get_num_fine_lemmas(self):
#         return self.train_dataset.get_num_fine_lemmas()
    
#     def get_num_fine_coarse_senses(self):
#         return self.train_dataset.get_num_fine_coarse_senses()
    
#     def get_num_fine_coarse_pos_tags(self):
#         return self.train_dataset.get_num_fine
    
import torch
class Trainer:
    def __init__(self, train_dataloader, dev_dataloader):
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader


    def train_epoch(self):

        for words, pos_tags, senses in self.train_dataloader:
            continue

        return None
