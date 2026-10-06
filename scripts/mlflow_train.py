from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlflow,yaml
from src.trainer import train
cfg=yaml.safe_load(open("configs/tiny.yaml"));
with mlflow.start_run():
    mlflow.log_params(cfg); m=train(cfg); mlflow.log_metrics({k:v for k,v in m.items() if isinstance(v,(int,float))}); print(m)
