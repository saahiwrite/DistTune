import torch.nn as nn
class TinyTransformerLM(nn.Module):
    def __init__(self,vocab=256,d_model=64,nhead=4,layers=2):
        super().__init__(); self.emb=nn.Embedding(vocab,d_model); enc=nn.TransformerEncoderLayer(d_model,nhead,batch_first=True); self.net=nn.TransformerEncoder(enc,layers); self.head=nn.Linear(d_model,vocab)
    def forward(self,x): return self.head(self.net(self.emb(x)))
