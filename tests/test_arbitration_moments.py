"""MVHG moment gate for PHASE3 contested-cell allocation (spec 2026-06-21
arbitration-perf §5.1). PASSES on the exact current code; MUST stay green after
the urn-draw swap. A statistical bug here is silent -> it would poison the
recorded dataset, so this gate is mandatory, not optional."""
import torch
from des.kernels.arbitration import phase3_arbitrate_vec
from des.kernels.reproduction import ArrivalBuffer

DEV = torch.device("cpu")        # gate runs on CPU: deterministic, no GPU needed
MAXSID = 256
NFAC = 4


def _one_cell_arrivals(counts):
    """One contested cell at (0,0); record i = strain (i+1), faction 0, count c_i."""
    buf = ArrivalBuffer(DEV)
    for i, c in enumerate(counts):
        buf.add(torch.tensor([0]), torch.tensor([0]),
                torch.tensor([i + 1], dtype=torch.int32),
                torch.tensor([c], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int8))
    return buf.tensors()


def _survivors_per_record(counts, K, trials):
    """Run phase3 `trials` times into an empty K-slot cell; return [trials, R]
    survivor-count matrix keyed by the strain ids (i+1)."""
    R = len(counts)
    out = torch.zeros((trials, R), dtype=torch.float64)
    for seed in range(trials):
        sid = torch.zeros((1, 1, K), dtype=torch.int32)
        cnt = torch.zeros((1, 1, K), dtype=torch.int32)
        fac = torch.zeros((1, 1, K), dtype=torch.int8)
        birth = torch.zeros((1, 1, K), dtype=torch.int32)
        arr = _one_cell_arrivals(counts)
        g = torch.Generator(device=DEV); g.manual_seed(seed)
        nsid, ncnt, _, _ = phase3_arbitrate_vec(
            sid, cnt, fac, arr, K=K, birth_tick=birth, T=1, generator=g,
            MAXSID=MAXSID, NFAC=NFAC)
        seated = int(ncnt[0, 0].sum())
        assert seated == K, f"seed={seed}: seated {seated} != avail {K}"  # hard cap
        for i in range(R):
            m = nsid[0, 0] == (i + 1)
            out[seed, i] = int(ncnt[0, 0][m].sum())
    return out


def test_mvhg_proportional_means_hot_regime():
    # N=1000, avail=10. MVHG mean_i = avail * c_i / N = [1, 2, 7].
    counts = [100, 200, 700]; K = 10; N = sum(counts)
    surv = _survivors_per_record(counts, K, trials=10000)
    means = surv.mean(dim=0)
    expected = torch.tensor([K * c / N for c in counts], dtype=torch.float64)
    assert torch.allclose(means, expected, atol=0.15), \
        f"means {means.tolist()} != MVHG {expected.tolist()}"
    # every trial seats exactly avail across records
    assert torch.all(surv.sum(dim=1) == K)


def test_mvhg_without_replacement_variance_small_N():
    # N=9, avail=6, counts [3,3,3]. MVHG var_i = avail*p*(1-p)*(N-avail)/(N-1)
    #   = 6*(1/3)*(2/3)*(3/8) = 0.5  ;  multinomial (with-replacement) var = 1.333.
    # The exact draw must match 0.5, NOT 1.333 -> this discriminates exact vs approx.
    counts = [3, 3, 3]; K = 6; N = sum(counts)
    surv = _survivors_per_record(counts, K, trials=10000)
    var = surv.var(dim=0, unbiased=True)
    p = 1.0 / 3.0
    mvhg_var = K * p * (1 - p) * (N - K) / (N - 1)          # 0.5
    multinomial_var = K * p * (1 - p)                        # 1.333
    assert torch.all(var < 0.5 * (mvhg_var + multinomial_var)), \
        f"variance {var.tolist()} looks with-replacement (MVHG={mvhg_var:.3f}, " \
        f"multinomial={multinomial_var:.3f})"
    means = surv.mean(dim=0)
    assert torch.allclose(means, torch.full((3,), 2.0, dtype=torch.float64),
                          atol=0.1), f"means {means.tolist()} != [2,2,2]"
