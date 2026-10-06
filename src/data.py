import torch
from torch.utils.data import Dataset
class RandomTokenDataset(Dataset):
    def __init__(self,n=512,seq=32,vocab=256,seed=7):
        g=torch.Generator().manual_seed(seed); self.x=torch.randint(0,vocab,(n,seq+1),generator=g)
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i,:-1],self.x[i,1:]
