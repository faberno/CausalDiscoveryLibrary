from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from cdl.data.sachs import sachs

class SachsDataModule(LightningDataModule):
    def __init__(self, batch_size=None, obs=True):
        super().__init__()
        self.batch_size = batch_size
        self.obs = obs
        self.data = None
        self.graph = None

    def setup(self, stage: str) -> None:
        G, X = sachs(obs=self.obs, as_df=False)
        self.graph = G
        self.data = X
        if self.batch_size is None:
            self.batch_size = len(self.data)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data)