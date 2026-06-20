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


def test_kwall_order_independent():
    """
    3 strains with EQUAL arrivals (60/60/60) into an empty cell with K=8 (available=8).
    Heavily oversubscribed: total=180 >> available=8.

    Asserts:
      (a) every trial seats EXACTLY 8 new individuals (hard cap, exact)
      (b) all three strains' mean survivors ≈ equal (each ≈ 8/3 ≈ 2.667),
          pairwise max/min ratio < 1.10
          (old sequential binom code gives ~1.24 → this is the regression gate)
      (c) same means hold when arrivals are enumerated in REVERSE order
          (proves enumeration-order independence directly)
    """
    K = 8
    TRIALS = 3000
    STRAINS = [1, 2, 3]
    ARRIVAL = 60  # equal, heavily oversubscribed

    def run_trials(strain_order):
        accum = {s: 0 for s in STRAINS}
        for seed in range(TRIALS):
            sid   = torch.zeros((1, 1, K), dtype=torch.int32)
            cnt   = torch.zeros((1, 1, K), dtype=torch.int32)
            birth = torch.zeros((1, 1, K), dtype=torch.int32)
            # cell is empty: resident_occ=0, available=K=8
            arr = _arrivals([(0, 0, s, ARRIVAL) for s in strain_order])
            g = torch.Generator(device=DEV); g.manual_seed(seed)
            nsid, ncnt, _ = phase3_arbitrate(
                sid, cnt, arr, K=K, birth_tick=birth, T=1, generator=g, MAXSID=MAXSID
            )
            total_seated = int(ncnt[0, 0].sum())
            # (a) hard cap: exactly available=8 seated every trial
            assert total_seated == K, \
                f"seed={seed} order={strain_order}: seated {total_seated} != {K}"
            for s in STRAINS:
                m = nsid[0, 0] == s
                accum[s] += int(ncnt[0, 0][m].sum())
        return accum

    # Forward order
    accum_fwd = run_trials(STRAINS)
    means_fwd = [accum_fwd[s] / TRIALS for s in STRAINS]
    ratio_fwd = max(means_fwd) / min(means_fwd)
    # (b) pairwise max/min ratio < 1.10  (old code: ~1.24)
    assert ratio_fwd < 1.10, \
        f"order-bias detected (fwd): means={[f'{m:.3f}' for m in means_fwd]}, ratio={ratio_fwd:.4f}"

    # (c) reverse order — means must be unchanged within tolerance
    accum_rev = run_trials(list(reversed(STRAINS)))
    means_rev = [accum_rev[s] / TRIALS for s in STRAINS]
    ratio_rev = max(means_rev) / min(means_rev)
    assert ratio_rev < 1.10, \
        f"order-bias detected (rev): means={[f'{m:.3f}' for m in means_rev]}, ratio={ratio_rev:.4f}"
    # cross-order agreement: each strain's mean should agree within 5% between fwd/rev
    for s in STRAINS:
        mf = accum_fwd[s] / TRIALS
        mr = accum_rev[s] / TRIALS
        cross_ratio = max(mf, mr) / max(min(mf, mr), 1e-9)
        assert cross_ratio < 1.05, \
            f"strain {s}: fwd_mean={mf:.3f} rev_mean={mr:.3f} cross_ratio={cross_ratio:.4f}"
