import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self,train,val,test, batch_size: int):
        super().__init__()
        self.train_datasets = train
        self.val_datasets = val
        self.test_datasets = test
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(self.train_datasets,batch_size=self.batch_size,shuffle=True,num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.val_datasets,batch_size=self.batch_size,num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.test_datasets,batch_size=self.batch_size,num_workers=24)