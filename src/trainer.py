import time, torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .data import RandomTokenDataset
from .model import TinyTransformerLM
from .distributed import setup,cleanup

def train(cfg):
    rank,world,local=setup(); device=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    ds=RandomTokenDataset(cfg.get("samples",256),cfg.get("seq",32)); sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True) if world>1 else None
    dl=DataLoader(ds,batch_size=cfg.get("batch_size",16),sampler=sampler,shuffle=sampler is None)
    model=TinyTransformerLM().to(device); model=DDP(model,device_ids=[local] if torch.cuda.is_available() else None) if world>1 else model
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.get("lr",3e-4)); loss_fn=torch.nn.CrossEntropyLoss(); steps=0; start=time.perf_counter(); last=0
    for epoch in range(cfg.get("epochs",1)):
        if sampler:sampler.set_epoch(epoch)
        for x,y in dl:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits.reshape(-1,logits.size(-1)),y.reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); last=float(loss.detach()); steps+=1
    elapsed=time.perf_counter()-start; metrics={"rank":rank,"world_size":world,"steps":steps,"loss":last,"elapsed_s":elapsed,"steps_per_s":steps/max(elapsed,1e-9)}; cleanup(); return metrics
