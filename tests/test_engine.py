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

def test_fixation_is_single_faction_field():
    # hand-build an engine state where only one faction survives -> _fixated True
    e = Engine(H=8, W=8, K=16, seed=0, device=DEV)
    e.world.count.zero_(); e.world.strain_id.zero_(); e.world.faction.zero_()
    e.world.strain_id[0, 0, 0] = 1; e.world.count[0, 0, 0] = 5; e.world.faction[0, 0, 0] = 2
    e.world.strain_id[1, 1, 0] = 1; e.world.count[1, 1, 0] = 5; e.world.faction[1, 1, 0] = 2
    assert e._fixated() is True            # all survivors faction 2
    e.world.faction[1, 1, 0] = 3           # introduce a second faction
    assert e._fixated() is False

def test_engine_seeds_four_factions():
    e = Engine(H=16, W=16, K=32, seed=0, device=DEV)
    live = e.world.count.sum(dim=-1) > 0
    assert int(live.sum()) == 4            # four quadrant centers seeded
    facs = e.world.faction[e.world.count > 0]
    assert set(facs.tolist()) == {0, 1, 2, 3}
