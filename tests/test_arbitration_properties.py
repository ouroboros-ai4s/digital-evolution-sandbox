# Property tests for vectorized arbitration (replaces the deleted bit-identical
# equivalence suite). Asserts the invariants the random-key K-wall must hold.
from __future__ import annotations
import torch
from des.kernels.arbitration import phase3_arbitrate_vec
from des.kernels.reproduction import ArrivalBuffer

DEV = torch.device("cpu")
MAXSID = 64
NFAC = 4


def _arrivals(events):
    """events: list of (y, x, sid, cnt, fac)."""
    buf = ArrivalBuffer(DEV)
    for (y, x, sid, cnt, fac) in events:
        buf.add(torch.tensor([y]), torch.tensor([x]),
                torch.tensor([sid], dtype=torch.int32),
                torch.tensor([cnt], dtype=torch.int32),
                torch.tensor([fac], dtype=torch.int8))
    return buf.tensors()


def _empty_state(H, W, K):
    z = lambda dt: torch.zeros((H, W, K), dtype=dt, device=DEV)
    return z(torch.int32), z(torch.int32), z(torch.int8), z(torch.int32)


def test_hard_cap_never_exceeds_K():
    K = 8
    sid, cnt, fac, birth = _empty_state(1, 1, K)
    arr = _arrivals([(0, 0, 5, 100, 0), (0, 0, 7, 100, 1)])  # 200 into empty cell
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
    assert int(ncnt[0, 0].sum()) == K            # exactly avail=K seated

def test_conservation_uncontested_all_survive():
    K = 64
    sid, cnt, fac, birth = _empty_state(1, 1, K)
    arr = _arrivals([(0, 0, 5, 10, 0), (0, 0, 7, 20, 1)])   # 30 << 64, no thinning
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nfac, _ = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
    seated = {(int(nsid[0,0,k]), int(nfac[0,0,k])): int(ncnt[0,0,k])
              for k in range(K) if ncnt[0,0,k] > 0}
    assert seated == {(5, 0): 10, (7, 1): 20}

def test_same_sid_different_faction_kept_separate():
    # red/blue BB0 (same sid, faction 0 vs 1) must NOT merge into one slot.
    K = 64
    sid, cnt, fac, birth = _empty_state(1, 1, K)
    arr = _arrivals([(0, 0, 5, 10, 0), (0, 0, 5, 15, 1)])   # same sid, diff faction
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nfac, _ = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
    seated = {(int(nsid[0,0,k]), int(nfac[0,0,k])): int(ncnt[0,0,k])
              for k in range(K) if ncnt[0,0,k] > 0}
    assert seated == {(5, 0): 10, (5, 1): 15}    # two distinct slots

def test_faction_blind_equal_arrivals():
    # equal arrivals from two factions, heavily oversubscribed: survivors ~equal.
    K = 8
    accum = {0: 0, 1: 0}
    for seed in range(400):
        sid, cnt, fac, birth = _empty_state(1, 1, K)
        arr = _arrivals([(0, 0, 5, 100, 0), (0, 0, 7, 100, 1)])
        g = torch.Generator(device=DEV); g.manual_seed(seed)
        nsid, ncnt, nfac, _ = phase3_arbitrate_vec(
            sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
        for k in range(K):
            if ncnt[0,0,k] > 0:
                accum[int(nfac[0,0,k])] += int(ncnt[0,0,k])
    ratio = accum[0] / max(1, accum[1])
    assert 0.85 < ratio < 1.18, f"faction-blind violated: {accum}, ratio={ratio:.3f}"

def test_resident_not_evicted():
    K = 8
    sid, cnt, fac, birth = _empty_state(1, 1, K)
    sid[0,0,0] = 9; cnt[0,0,0] = 5; fac[0,0,0] = 2          # resident: avail = 3
    arr = _arrivals([(0, 0, 5, 100, 0), (0, 0, 7, 100, 1)])
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nfac, _ = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
    res = (nsid[0,0] == 9) & (nfac[0,0] == 2)
    assert int(ncnt[0,0][res].sum()) == 5                   # resident intact
    assert int(ncnt[0,0].sum()) == 8                        # 5 resident + 3 new
