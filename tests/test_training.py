from src.trainer import train
def test_single_process_training_runs():
    m=train({"epochs":1,"samples":32,"seq":8,"batch_size":8,"lr":1e-3}); assert m["steps"]>0; assert m["loss"]>0; assert m["world_size"]==1
