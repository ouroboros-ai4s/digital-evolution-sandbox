# tests/test_engine.py
import torch
from des.engine import Engine

DEV = torch.device("cpu")

def test_engine_runs_without_error():
    e = Engine(H=16, W=16, K=32, seed=1, device=DEV)
    ran = e.run(ticks=10)
    assert ran >= 1
    assert e.total_count() >= 0

def test_determinism_same_seed():
    e1 = Engine(H=16, W=16, K=32, seed=42, device=DEV); e1.run(15)
    e2 = Engine(H=16, W=16, K=32, seed=42, device=DEV); e2.run(15)
    assert torch.equal(e1.world.count, e2.world.count)
    assert torch.equal(e1.world.strain_id, e2.world.strain_id)

def test_different_seeds_diverge_or_vary():
    e1 = Engine(H=16, W=16, K=32, seed=1, device=DEV); e1.run(20)
    e2 = Engine(H=16, W=16, K=32, seed=2, device=DEV); e2.run(20)
    # at least one of count/strain tensors differs (stochasticity is live)
    assert (not torch.equal(e1.world.count, e2.world.count)) or \
           (not torch.equal(e1.world.strain_id, e2.world.strain_id))

def test_not_instant_extinction():
    e = Engine(H=16, W=16, K=32, seed=7, device=DEV)
    e.run(5)
    assert e.total_count() > 0          # world is not dead after a few ticks
