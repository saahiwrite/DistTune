import argparse,json,yaml
from src.trainer import train
p=argparse.ArgumentParser();p.add_argument("--config",default="configs/tiny.yaml");a=p.parse_args(); cfg=yaml.safe_load(open(a.config)); m=train(cfg); print(json.dumps(m,indent=2))
