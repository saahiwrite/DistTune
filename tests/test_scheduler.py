from src.scheduler import *
def test_batch(): assert per_rank_batch(TrainConfig(4,64))==16
def test_speedup(): assert speedup(100,65)==35.0
def test_grid(): assert len(trial_grid())==4
