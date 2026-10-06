from dataclasses import dataclass
@dataclass
class TrainConfig: world_size:int=1; global_batch:int=32; epochs:int=3
def per_rank_batch(c):
 if c.world_size<1 or c.global_batch%c.world_size: raise ValueError('global_batch must divide world_size')
 return c.global_batch//c.world_size
def speedup(single_seconds,distributed_seconds): return round((single_seconds-distributed_seconds)/single_seconds*100,2)
def trial_grid(lrs=(1e-4,3e-4),dropouts=(0.0,0.1)):
 return [{'lr':lr,'dropout':d} for lr in lrs for d in dropouts]
