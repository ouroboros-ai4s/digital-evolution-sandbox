# tests/test_arbitration.py
import torch
from des.kernels.arbitration import phase3_arbitrate
from des.kernels.reproduction import ArrivalBuffer

DEV = torch.device("cpu")
MAXSID = 64


def _arrivals(events):
    buf = ArrivalBuffer(DEV)
    for (y, x, sid, cnt) in events:
        buf.add(torch.tensor([y]), torch.tensor([x]),
                torch.tensor([sid], dtype=torch.int32), torch.tensor([cnt], dtype=torch.int32))
    return buf.tensors()


def test_arrivals_seat_into_empty_cell():
    # 30 total individuals arrive (strain5:10 + strain7:20) into an empty cell.
    # K=64 >> 30 → no thinning → full counts seat unchanged.
    K = 64
    sid   = torch.zeros((1, 1, K), dtype=torch.int32)
    cnt   = torch.zeros((1, 1, K), dtype=torch.int32)
    birth = torch.zeros((1, 1, K), dtype=torch.int32)
    arr = _arrivals([(0, 0, 5, 10), (0, 0, 7, 20)])
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nbirth = phase3_arbitrate(
        sid, cnt, arr, K=K, birth_tick=birth, T=3, generator=g, MAXSID=MAXSID
    )
    seated = {int(nsid[0, 0, k]): int(ncnt[0, 0, k])
              for k in range(K) if ncnt[0, 0, k] > 0}
    assert seated == {5: 10, 7: 20}
    # new strains in an empty cell → all occupied slots stamped with birth tick T=3
    for k in range(K):
        if ncnt[0, 0, k] > 0:
            assert int(nbirth[0, 0, k]) == 3


def test_convergent_arrivals_merge_same_strain():
    # Same strain arrives in two separate events (10 + 15 = 25 total individuals).
    # K=64 >> 25 → no thinning → coalesced into one slot with count 25.
    K = 64
    sid   = torch.zeros((1, 1, K), dtype=torch.int32)
    cnt   = torch.zeros((1, 1, K), dtype=torch.int32)
    birth = torch.zeros((1, 1, K), dtype=torch.int32)
    arr = _arrivals([(0, 0, 5, 10), (0, 0, 5, 15)])    # same strain twice
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, _ = phase3_arbitrate(
        sid, cnt, arr, K=K, birth_tick=birth, T=1, generator=g, MAXSID=MAXSID
    )
    seated = {int(nsid[0, 0, k]): int(ncnt[0, 0, k])
              for k in range(K) if ncnt[0, 0, k] > 0}
    assert seated == {5: 25}                            # merged into one slot


def test_kwall_thins_when_over_capacity():
    # K=8, resident strain9 holds 5 individuals → available=3.
    # Strains 5 and 7 each arrive with 100 individuals (total 200 >> 3).
    # Statistical loop over many seeds proves:
    #   (a) new seated individuals per trial ≤ available (3)
    #   (b) resident strain9 never evicted
    #   (c) strain-blind: equal arrivals → accumulated survivors within ±15%
    K = 8
    TRIALS = 200
    accum = {5: 0, 7: 0}

    for seed in range(TRIALS):
        sid   = torch.zeros((1, 1, K), dtype=torch.int32)
        cnt   = torch.zeros((1, 1, K), dtype=torch.int32)
        birth = torch.zeros((1, 1, K), dtype=torch.int32)
        cnt[0, 0, 0] = 5
        sid[0, 0, 0] = 9                                # resident: 5 individuals of strain9
        arr = _arrivals([(0, 0, 5, 100), (0, 0, 7, 100)])  # equal arrivals, total 200
        g = torch.Generator(device=DEV); g.manual_seed(seed)
        nsid, ncnt, _ = phase3_arbitrate(
            sid, cnt, arr, K=K, birth_tick=birth, T=1, generator=g, MAXSID=MAXSID
        )

        # (a) total new individuals seated ≤ available (3); residents already there
        new_total = int(ncnt[0, 0].sum()) - 5           # subtract resident count
        assert new_total <= 3, \
            f"seed={seed}: new seated {new_total} > available 3"

        # (b) resident strain9 never evicted
        res_mask = nsid[0, 0] == 9
        assert int(ncnt[0, 0][res_mask].sum()) == 5, \
            f"seed={seed}: resident strain9 evicted"

        # accumulate for strain-blind check
        for s in (5, 7):
            m = nsid[0, 0] == s
            accum[s] += int(ncnt[0, 0][m].sum())

    # (c) strain-blind: equal arrivals → ratio of accumulated survivors within 15%
    total5, total7 = accum[5], accum[7]
    # avoid division by zero in degenerate all-zero case
    if total5 + total7 > 0:
        ratio = total5 / max(total7, 1) if total7 > 0 else float("inf")
        assert 0.85 < ratio < 1.18, \
            f"strain-blind violation: strain5={total5}, strain7={total7}, ratio={ratio:.3f}"
