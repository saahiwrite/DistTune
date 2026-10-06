import os, torch, torch.distributed as dist

def setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo"); rank=dist.get_rank(); local=int(os.environ.get("LOCAL_RANK",0));
        if torch.cuda.is_available(): torch.cuda.set_device(local)
        return rank,dist.get_world_size(),local
    return 0,1,0

def cleanup():
    if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()
