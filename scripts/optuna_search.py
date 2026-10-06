from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optuna
from src.trainer import train
def objective(trial):
    m=train({"epochs":1,"samples":96,"seq":16,"batch_size":trial.suggest_categorical("batch_size",[8,16,32]),"lr":trial.suggest_float("lr",1e-4,1e-3,log=True)}); return m["loss"]
study=optuna.create_study(direction="minimize");study.optimize(objective,n_trials=5);print(study.best_params,study.best_value)
