# tests/test_arbitration_vec_equiv.py
# Equivalence gate: phase3_arbitrate_vec must produce identical per-cell
# {sid: (count, birth)} dicts as the reference phase3_arbitrate for every
# seed.  Slot ORDER is irrelevant; only the dict contents are compared.
from __future__ import annotations
import pytest
import torch
from des.kernels.arbitration import phase3_arbitrate, phase3_arbitrate_vec

H, W, K = 8, 8, 16
MAXSID   = 64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cell_dict(sid_hwk, cnt_hwk, birth_hwk):
    """Return {(y,x): {sid: (count, birth)}} for occupied slots only."""
    result = {}
    H, W, K = sid_hwk.shape
    for y in range(H):
        for x in range(W):
            d = {}
            for k in range(K):
                c = int(cnt_hwk[y, x, k])
                if c > 0:
                    s = int(sid_hwk[y, x, k])
                    b = int(birth_hwk[y, x, k])
                    assert s not in d, f"duplicate sid {s} at ({y},{x}) in slot {k}"
                    d[s] = (c, b)
            if d:
                result[(y, x)] = d
    return result


def _rand_residents(rng: torch.Generator, dev: torch.device):
    """Random resident state: ~0..K/3 occupancy per cell, sids 1..50, counts 1..8, birth 0..5."""
    sid   = torch.zeros((H, W, K), dtype=torch.int32,  device=dev)
    cnt   = torch.zeros((H, W, K), dtype=torch.int32,  device=dev)
    birth = torch.zeros((H, W, K), dtype=torch.int32,  device=dev)
    for y in range(H):
        for x in range(W):
            # random number of occupied slots: 0..K//3
            n_occ = int(torch.randint(0, K // 3 + 1, (1,), generator=rng).item())
            if n_occ == 0:
                continue
            sids_used = set()
            for k in range(n_occ):
                while True:
                    s = int(torch.randint(1, 51, (1,), generator=rng).item())
                    if s not in sids_used:
                        break
                sids_used.add(s)
                sid[y, x, k]   = s
                cnt[y, x, k]   = int(torch.randint(1, 9, (1,), generator=rng).item())
                birth[y, x, k] = int(torch.randint(0, 6, (1,), generator=rng).item())
    return sid, cnt, birth


def _rand_arrivals(rng: torch.Generator, dev: torch.device, n_events: int = 40):
    """Random arrivals: sids 1..60 (may overlap residents), counts 1..5."""
    if n_events == 0:
        empty = torch.zeros(0, dtype=torch.long, device=dev)
        return (empty, empty,
                torch.zeros(0, dtype=torch.int32, device=dev),
                torch.zeros(0, dtype=torch.int32, device=dev))
    ys   = torch.randint(0, H, (n_events,), generator=rng, device=dev).long()
    xs   = torch.randint(0, W, (n_events,), generator=rng, device=dev).long()
    sids = torch.randint(1, 61, (n_events,), generator=rng, device=dev).to(torch.int32)
    cnts = torch.randint(1, 6,  (n_events,), generator=rng, device=dev).to(torch.int32)
    return ys, xs, sids, cnts


def _run_both(seed: int, dev: torch.device,
              sid, cnt, birth, arrivals):
    """Run both functions with freshly-seeded identical generators; return cell dicts."""
    g_ref = torch.Generator(device=dev); g_ref.manual_seed(seed)
    g_vec = torch.Generator(device=dev); g_vec.manual_seed(seed)

    r_sid, r_cnt, r_birth = phase3_arbitrate(
        sid.clone(), cnt.clone(), arrivals, K,
        birth.clone(), T=seed % 20, generator=g_ref, MAXSID=MAXSID)

    v_sid, v_cnt, v_birth = phase3_arbitrate_vec(
        sid.clone(), cnt.clone(), arrivals, K,
        birth.clone(), T=seed % 20, generator=g_vec, MAXSID=MAXSID)

    return (_cell_dict(r_sid, r_cnt, r_birth),
            _cell_dict(v_sid, v_cnt, v_birth))


# ---------------------------------------------------------------------------
# main random-seed equivalence sweep (CPU)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(200))
def test_equiv_cpu_random(seed):
    dev  = torch.device("cpu")
    rng  = torch.Generator(device=dev); rng.manual_seed(seed + 10000)
    sid, cnt, birth = _rand_residents(rng, dev)
    arrivals        = _rand_arrivals(rng, dev, n_events=40)

    ref_d, vec_d = _run_both(seed, dev, sid, cnt, birth, arrivals)
    assert ref_d == vec_d, (
        f"seed={seed} CPU: dicts differ\n  ref={ref_d}\n  vec={vec_d}")


# ---------------------------------------------------------------------------
# CUDA sweep (~20 seeds) -- skipped if no CUDA
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("seed", range(20))
def test_equiv_cuda_random(seed):
    dev  = torch.device("cuda")
    rng  = torch.Generator(device="cpu"); rng.manual_seed(seed + 20000)
    sid_cpu, cnt_cpu, birth_cpu = _rand_residents(rng, torch.device("cpu"))
    arr_cpu = _rand_arrivals(rng, torch.device("cpu"), n_events=40)

    sid   = sid_cpu.to(dev)
    cnt   = cnt_cpu.to(dev)
    birth = birth_cpu.to(dev)
    arrivals = tuple(t.to(dev) for t in arr_cpu)

    ref_d, vec_d = _run_both(seed, dev, sid, cnt, birth, arrivals)
    assert ref_d == vec_d, (
        f"seed={seed} CUDA: dicts differ\n  ref={ref_d}\n  vec={vec_d}")


# ---------------------------------------------------------------------------
# explicit edge cases
# ---------------------------------------------------------------------------

def _make_empty_residents(dev):
    sid   = torch.zeros((H, W, K), dtype=torch.int32,  device=dev)
    cnt   = torch.zeros((H, W, K), dtype=torch.int32,  device=dev)
    birth = torch.zeros((H, W, K), dtype=torch.int32,  device=dev)
    return sid, cnt, birth


def test_edge_empty_arrivals():
    """No arrivals -> both return resident state unchanged."""
    dev = torch.device("cpu")
    sid, cnt, birth = _make_empty_residents(dev)
    cnt[0, 0, 0] = 3; sid[0, 0, 0] = 7  # one resident
    empty = (torch.zeros(0, dtype=torch.long, device=dev),
             torch.zeros(0, dtype=torch.long, device=dev),
             torch.zeros(0, dtype=torch.int32, device=dev),
             torch.zeros(0, dtype=torch.int32, device=dev))
    ref_d, vec_d = _run_both(0, dev, sid, cnt, birth, empty)
    assert ref_d == vec_d


def test_edge_all_resident():
    """All arriving strains already occupy slots -> case-a only path."""
    dev = torch.device("cpu")
    sid, cnt, birth = _make_empty_residents(dev)
    # place strains 1,2,3 in cell (0,0)
    for k, s in enumerate([1, 2, 3]):
        sid[0, 0, k] = s; cnt[0, 0, k] = 2; birth[0, 0, k] = 1
    # arrivals: same three strains, small counts so no K-wall trigger
    ys   = torch.tensor([0, 0, 0], dtype=torch.long,  device=dev)
    xs   = torch.tensor([0, 0, 0], dtype=torch.long,  device=dev)
    sids = torch.tensor([1, 2, 3], dtype=torch.int32, device=dev)
    cnts = torch.tensor([1, 1, 1], dtype=torch.int32, device=dev)
    ref_d, vec_d = _run_both(42, dev, sid, cnt, birth, (ys, xs, sids, cnts))
    assert ref_d == vec_d


def test_edge_all_new():
    """All arriving strains are new (empty cell) -> case-b only path."""
    dev = torch.device("cpu")
    sid, cnt, birth = _make_empty_residents(dev)
    ys   = torch.tensor([3, 3, 3], dtype=torch.long,  device=dev)
    xs   = torch.tensor([5, 5, 5], dtype=torch.long,  device=dev)
    sids = torch.tensor([10, 20, 30], dtype=torch.int32, device=dev)
    cnts = torch.tensor([2, 3, 1],   dtype=torch.int32, device=dev)
    ref_d, vec_d = _run_both(7, dev, sid, cnt, birth, (ys, xs, sids, cnts))
    assert ref_d == vec_d


def test_edge_cell_at_capacity():
    """Cell has exactly K slots filled -> newcomers get zeroed by K-wall."""
    dev = torch.device("cpu")
    sid, cnt, birth = _make_empty_residents(dev)
    # fill cell (2,2) to full capacity K=16
    for k in range(K):
        sid[2, 2, k] = k + 1; cnt[2, 2, k] = 1
    ys   = torch.tensor([2], dtype=torch.long,  device=dev)
    xs   = torch.tensor([2], dtype=torch.long,  device=dev)
    sids = torch.tensor([99], dtype=torch.int32, device=dev)
    cnts = torch.tensor([5],  dtype=torch.int32, device=dev)
    ref_d, vec_d = _run_both(3, dev, sid, cnt, birth, (ys, xs, sids, cnts))
    assert ref_d == vec_d


def test_edge_multiple_new_strains_same_cell():
    """Multiple NEW strains arrive in the same cell -> each must land in a distinct slot.
    Note: sids must be < MAXSID=64 to avoid key-encoding overflow in section 1."""
    dev = torch.device("cpu")
    sid, cnt, birth = _make_empty_residents(dev)
    n = 6  # 6 new strains into empty cell (2,3)
    ys   = torch.full((n,), 2, dtype=torch.long,  device=dev)
    xs   = torch.full((n,), 3, dtype=torch.long,  device=dev)
    # use sids 10..15 (all < MAXSID=64) so key encoding is non-overlapping
    sids = torch.arange(10, 10 + n, dtype=torch.int32, device=dev)
    cnts = torch.ones(n, dtype=torch.int32, device=dev)
    ref_d, vec_d = _run_both(11, dev, sid, cnt, birth, (ys, xs, sids, cnts))
    assert ref_d == vec_d
    # additionally confirm no duplicate sid in vec output and all 6 seated
    cell = vec_d.get((2, 3), {})
    assert len(cell) == n, f"expected {n} distinct sids, got {cell}"


def test_edge_strain_both_resident_and_arriving():
    """A strain that already occupies a slot also arrives -> counts add (case-a)."""
    dev = torch.device("cpu")
    sid, cnt, birth = _make_empty_residents(dev)
    sid[1, 1, 0] = 5; cnt[1, 1, 0] = 3; birth[1, 1, 0] = 0  # resident strain 5
    ys   = torch.tensor([1], dtype=torch.long,  device=dev)
    xs   = torch.tensor([1], dtype=torch.long,  device=dev)
    sids = torch.tensor([5], dtype=torch.int32, device=dev)
    cnts = torch.tensor([4], dtype=torch.int32, device=dev)
    ref_d, vec_d = _run_both(99, dev, sid, cnt, birth, (ys, xs, sids, cnts))
    assert ref_d == vec_d
    # strain 5 should have count 3+4=7
    assert vec_d[(1, 1)][5][0] == 7


def test_edge_no_duplicate_sid_per_cell_vec():
    """Run vec over 50 random seeds and assert no cell ever holds duplicate sids."""
    dev = torch.device("cpu")
    for seed in range(50):
        rng = torch.Generator(device=dev); rng.manual_seed(seed + 30000)
        sid, cnt, birth = _rand_residents(rng, dev)
        arrivals        = _rand_arrivals(rng, dev, n_events=30)
        g_vec = torch.Generator(device=dev); g_vec.manual_seed(seed)
        v_sid, v_cnt, v_birth = phase3_arbitrate_vec(
            sid.clone(), cnt.clone(), arrivals, K,
            birth.clone(), T=5, generator=g_vec, MAXSID=MAXSID)
        for y in range(H):
            for x in range(W):
                sids_occ = [int(v_sid[y, x, k])
                            for k in range(K) if v_cnt[y, x, k] > 0]
                assert len(sids_occ) == len(set(sids_occ)), \
                    f"seed={seed}: duplicate sid at ({y},{x}): {sids_occ}"
