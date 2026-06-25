# S2 — 塑形突变谱 (shaped mutation spectrum) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `_spectrum_for(letter)` 从「单一 affinity 公式」升级为「读 `SPECTRUM_SHAPE` 表的三旋钮 (power / family_mask / flatten_mix) 塑形函数」,并把 P 池余下的 10 个突变基元(P_aic / P_ep / P_fscan / P_zscan / P_entropy_brake / P_loopswap_lite / P_neutral_sink / P_slow_drift / P_burst_lite / P_balanced)按 roster 加入 registry —— 让 P 池 12 个基元全部活在结构化的 shape 表里,zero 内核改动。

**Architecture:** 三件事,顺序: (1) `SPECTRUM_SHAPE: dict[str, tuple[float, FamilyMask, float]]` 表加入 `registry.py`,默认 `(1.0, None, 0.0)` 等价于「power=1 / 全 mask / 无 flatten」即 P_base 的现行行为; (2) `_spectrum_for(letter)` 在 S6 已经做完 gran-match + 等长预过滤的基础上,再读 shape 表,按 `w(t) = aff(src,fam(t))**power · 𝟙[mask(t)] ; w(t) = (1-mix)·w + mix·1/(|A|-1)` 套三旋钮,再归一; (3) `_P` 表加 10 个新行 + `SPECTRUM_SHAPE` 加 12 个对应键(含 5 个老 P 行,因为 P_base/P_hotspot 的 shape 也用同一表读),`ALPHABET` 加 10 个新键(全部 family `"P"`,gran `"residue"`)。无内核改动,无 `phenotype()` 改动(只是 `dominant_p` 现在能解析到新基元),无 `phenotype_arrays.py` 改动。

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest. Windows 主机, `PYTHONPATH=src` 纪律。引擎源码 `src/des/`。**依赖**: S0 (CLI 已就位,不动) + S6 (`GRAN` / `MOTIF_LEN` / gran-match 的 `_spectrum_for` / 预 filter 已落地,本 spec 在它上面叠 shape) + S1 (vis 通道,与谱形塑无交互,本 spec 不动它)。

## Global Constraints

- **谱形是结构函数**: 每条 shape `(power, family_mask, flatten_mix)` 都是 `(fam(x), fam(t), rank)` 的全局函数,**没有 per-species 量级**。shape biases mutation 的方向(toward F / toward N / sharpen / flatten),**绝不手写「这个株突变得更好」**(spec §2)。
- **`|A|` = full ALPHABET 大小**: P_ep 的 flatten 项分母 `1/(|A|-1)`,以及 `_spectrum_for` 的归一分母,都用 **live registry 的全 ALPHABET 字数**。**禁止子集 / 禁止排除「未铸造的字母」**(spec §2)。本 spec 落地时 ALPHABET 由 6 (S6/S1 末态) 增长到 16 (新增 10 P 行均为 family `"P"`,gran `"residue"`);S0/S6/S1 已经决定的「6→68 重录基线」纪律继续:`|A|` 跟着 registry 走,基线 fixtures 等到全 68 字铸造完才 RE-RECORD。
- **Default-run drift 是设计本身,不是 bug**(用户 2026-06-24 裁定 RE-RECORD): `|A|` 从 6 长到 16 时,**P_base 的归一分母变了 → 默认局每 tick 的突变分布会漂移**——这是设计兑现,不是 regression。**不要写「P_base/P_hotspot shape identical」这种断言**;只验「shape 旋钮按 roster 给出 bias」+「relabel-invariance」(spec §6)。byte-identity 锁守的是 **non-registry 代码路径**(`_mutation_outcomes` 块覆写、`feature_mask` predicate-bit、kernel match 关系),而不是 6-字母时代的具体数值。
- **`_spectrum_for` 是内核所见的唯一接口,签名不动**: 仍是 `_spectrum_for(letter: str) -> tuple[tuple[str, float], ...]`。S6 已经把 gran-match + 等长预过滤塞进函数里;S2 在那两个 predicate 之后再叠 `(power, family_mask, flatten_mix)` 三旋钮 + 归一,**但出口仍是同一个 `(target, q)` 序列**。`Phenotype.spectrum` cache、`_mutation_outcomes(seq, mutable, spectrum, blocks)` 调用点都不动。
- **三旋钮含义锁死,不发明新概念**(roster 直读): `power ∈ {1, 2, 3}` (1 = aff, 2 = aic 锐化, 3 = entropy_brake 极度锐化);`family_mask ∈ {None, {"F"}, {"Z"}, {"N"}, "adjacent"}` (`"adjacent"` = `|Δrank|=1`,`primitive-roster.md` line 100,P_loopswap_lite);`flatten_mix ∈ {0.0, 0.5}` (P_ep 唯一非零)。**12 个 P 基元全部用这三个旋钮覆盖,无 per-primitive 特殊路径**(spec §3 Lazy note)。
- **多 P 基元的合并归 S8**: 当 strain 同时携带多个 P 字母时,`phenotype()` 现行 `dominant_p` 规则(p_add 最大者,平手取第一次出现)**保留不变** —— S2 不解决多 P 混合,合并谱 `Σpᵢqᵢ/Σpᵢ` 是 S8 的活(spec §5 / §7,user 2026-06-24 裁定)。
- **frequency 表驱动: `T` (周期)、`p_add`、shape 三件事每条 P 行同行声明**, 不分散在三张表里。`_P[letter] = (p_add, period)` 仍然是 phenotype 的 P-magnitude 来源;`SPECTRUM_SHAPE[letter]` 是 shape 来源。`rate = min(p_max, μ + p_add)` 已实现,本 spec 不动 mutation rate 计算。
- **空谱守护已存在**: `family_mask` 排除掉所有 target(例如 P_fscan 在没有 F 字母的 registry 下)→ Σw=0 → `_spectrum_for` 返回 `()`,既有 `if tot==0: return ()` 直接接住(spec §5,无需新代码,只需一条测试守住)。
- **shape 旋钮值 module-load 校验**: `power ∈ {1, 2, 3}`,`family_mask ∈ {None, "F", "Z", "N", "adjacent"}`,`flatten_mix ∈ [0.0, 1.0]`。在 `SPECTRUM_SHAPE` 定义之后用 `assert` 守住值域;违反 → 模块 import 即 fail(spec §5)。
- **S5 `f-window` / S7 多位 / S8 A-pool 不在本 spec 触碰范围**: `P_burst_lite` roster 行写有 phase-modulation,**S5 owns f-window 基元**;S2 只把它当 `q ∝ aff` 的普通行写完(spec §7)。多 P 合并谱 `Σpᵢqᵢ/Σpᵢ` 归 S8(spec §7)。
- **回归锁**: S6 + S1 完工时全套 pytest 是绿的;S2 落地后,**non-spectrum 代码路径(kernel / phenotype_arrays / antagonism / reproduction kernel)的全部既有 pytest 必须维持绿**;只有「读 P_base spectrum 数值」的既有断言允许由用户 2026-06-24 RE-RECORD 裁定下进行更新——本 plan Task 7 显式列出预期受影响的测试名单与升级方式。

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/registry.py` | **Modify** | (a) `ALPHABET` 加 10 个新 P 字母(全 family `"P"`);(b) `GRAN` 配套加 10 行 `"residue"`;(c) `_P` 表加 10 行 `(p_add, period)`(每行 verbatim 抄 spec §1 表);(d) 新增 `SPECTRUM_SHAPE: dict[str, tuple[float, FamilyMask, float]]` 表覆盖全部 12 个 P 基元;(e) module-load 校验 `power ∈ {1,2,3}` / `family_mask ∈ {None,"F","Z","N","adjacent"}` / `flatten_mix ∈ [0,1]`;(f) `_spectrum_for(letter)` 在 S6 已有的 gran-match + 等长预过滤之后,叠加 shape 三旋钮(power / family_mask / flatten_mix)+ 重新归一。 |
| `tests/test_spectrum_shape.py` | **Create** | 新建,S2 owner 文件:`SPECTRUM_SHAPE` 覆盖 12 行 + 值域守门、各 shape 的 bias 验证(P_aic 比 P_base 锐 / P_ep 比 P_base 平 / P_entropy_brake 比 P_aic 更锐 / P_fscan 只在 F target / P_zscan 只在 Z target / P_neutral_sink 只在 N target / P_loopswap_lite 只在 |Δrank|=1 target / P_slow_drift / P_burst_lite / P_balanced 三者 shape ≡ P_base)、relabel-invariance、P_fscan 在无 F 字母 registry 下返回 `()`、Σq=1。 |
| `tests/test_registry.py` | **Modify (append)** | 追加 P 池 10 行 `_P` / `ALPHABET` / `GRAN` / `SPECTRUM_SHAPE` 在 registry 层的覆盖与值域断言(配合 S6 已有的「`GRAN` 覆盖每个 ALPHABET 字母」断言一并守住)。**不动**已有 `feature_mask` / `prey_mask` / `motif_blocks` 断言。 |
| `tests/test_phenotype_cache.py` | **Modify (append)** | 一条新断言:`phenotype(seq).spectrum` 是 `_spectrum_for(dominant_p)` 的 cache,本 spec 加完后 `dominant_p` 解析能落到任意一个新 P 字母上,phenotype 不抛异常。 |

**Naming contract (locked, used by every task):**

```python
# src/des/registry.py
FamilyMask = None | str          # None = no mask; "F"|"Z"|"N" = single-family mask;
                                 # "adjacent" = |Δrank|=1 (P_loopswap_lite)

SPECTRUM_SHAPE: dict[str, tuple[float, "FamilyMask", float]]
# letter -> (power, family_mask, flatten_mix)
# power      : 1.0 | 2.0 | 3.0
# family_mask: None | "F" | "Z" | "N" | "adjacent"
# flatten_mix: 0.0 (default) | 0.5 (P_ep)

# extended signature (body change only — signature unchanged from S6)
def _spectrum_for(letter: str) -> tuple[tuple[str, float], ...]
```

**P 池新增 10 行 verbatim 表(spec §1 + roster §P pool,**逐行用此表去填 `_P` / `SPECTRUM_SHAPE` / `ALPHABET` / `GRAN`,本 plan 后续步骤直接读这张表**):

| letter | family | gran | p_add | period | power | family_mask | flatten_mix | 注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `P_base` | P | residue | 0.0 | 1 | 1.0 | None | 0.0 | (S6/S0 已有,行不动) |
| `P_hotspot` | P | residue | 0.05 | 3 | 1.0 | None | 0.0 | (S6/S0 已有,行不动) |
| `P_aic` | P | residue | 0.03 | 3 | 2.0 | None | 0.0 | sharpen |
| `P_ep` | P | residue | 0.04 | 3 | 1.0 | None | 0.5 | flatten |
| `P_fscan` | P | residue | 0.02 | 5 | 1.0 | "F" | 0.0 | F-only |
| `P_zscan` | P | residue | 0.02 | 5 | 1.0 | "Z" | 0.0 | Z-only |
| `P_entropy_brake` | P | residue | 0.01 | 7 | 3.0 | None | 0.0 | super-sharpen |
| `P_loopswap_lite` | P | residue | 0.03 | 4 | 1.0 | "adjacent" | 0.0 | 仅 \|Δrank\|=1 |
| `P_neutral_sink` | P | residue | 0.02 | 5 | 1.0 | "N" | 0.0 | N-only |
| `P_slow_drift` | P | residue | 0.0 | 9 | 1.0 | None | 0.0 | 同 P_base shape,不同周期 |
| `P_burst_lite` | P | residue | 0.07 | 2 | 1.0 | None | 0.0 | 周期短,无 f-window;S5 owns |
| `P_balanced` | P | residue | 0.04 | 3 | 1.0 | None | 0.0 | 同 P_base shape,p_add=0.04 |

---

### Task 1: 把 10 行新 P 字母塞进 `ALPHABET` / `GRAN` / `_P`(纯数据扩张)

**Goal:** 把 spec §1 表里 10 个新 P 基元(`P_aic` / `P_ep` / `P_fscan` / `P_zscan` / `P_entropy_brake` / `P_loopswap_lite` / `P_neutral_sink` / `P_slow_drift` / `P_burst_lite` / `P_balanced`)的 family / gran / `(p_add, period)` 三件套写进 `registry.py`,**不**碰 `_spectrum_for` / 不碰 shape 表(Task 2/3 来)。这一步纯数据扩张:`ALPHABET` 从 6 增到 16,`GRAN` 跟着加 10 行 `"residue"`,`_P` 从 2 行增到 12 行。

**Files:**
- Modify: `src/des/registry.py:13-18` (`ALPHABET`) + `src/des/registry.py:34-37` (`_P`) + S6 加进来的 `GRAN` 块(本仓库里 S6 落地后 `GRAN` 紧跟 `ALPHABET`)
- Test: `tests/test_registry.py` (append)

**Interfaces:**
- Consumes: `ALPHABET` / `GRAN` / `_P` 的现行格式(S6 已加完 `GRAN`,S1 已加完 `VIS`)。
- Produces:
  - `ALPHABET["P_aic"] = "P"` ... 共 10 个新 key,全部 value `"P"`。
  - `GRAN["P_aic"] = "residue"` ... 共 10 个新 key,全部 value `"residue"`。
  - `_P["P_aic"] = (0.03, 3)` ... 共 10 行 `(p_add, period)`,数值 verbatim 从 File Structure 表读。

- [ ] **Step 1: 写失败测试 — 覆盖 + family 全 P + gran 全 residue + `(p_add, period)` 精确值**

追加到 `tests/test_registry.py`:

```python
# ---------------------------------------------------------------------------
# S2 Task 1: P pool expansion (10 new P primitives)
# ---------------------------------------------------------------------------

_S2_NEW_P = (
    ("P_aic",            0.03, 3),
    ("P_ep",             0.04, 3),
    ("P_fscan",          0.02, 5),
    ("P_zscan",          0.02, 5),
    ("P_entropy_brake",  0.01, 7),
    ("P_loopswap_lite",  0.03, 4),
    ("P_neutral_sink",   0.02, 5),
    ("P_slow_drift",     0.0,  9),
    ("P_burst_lite",     0.07, 2),
    ("P_balanced",       0.04, 3),
)


def test_s2_new_P_letters_present_in_alphabet_with_family_P():
    """每个新 P 字母在 ALPHABET 里, family 全是 'P'."""
    from des.registry import ALPHABET
    for letter, _, _ in _S2_NEW_P:
        assert letter in ALPHABET, f"{letter}: missing from ALPHABET"
        assert ALPHABET[letter] == "P", f"{letter}: family must be 'P', got {ALPHABET[letter]!r}"


def test_s2_new_P_letters_have_gran_residue():
    """所有新 P 字母 gran='residue' (motif P 基元归未来 spec, S2 不引入)."""
    from des.registry import GRAN
    for letter, _, _ in _S2_NEW_P:
        assert GRAN.get(letter) == "residue", f"{letter}: gran must be 'residue', got {GRAN.get(letter)!r}"


def test_s2_new_P_letters_have_exact_p_add_and_period():
    """每行 (p_add, period) 与 spec §1 表 verbatim 一致."""
    from des.registry import _P
    for letter, p_add, period in _S2_NEW_P:
        assert letter in _P, f"{letter}: missing from _P"
        assert _P[letter] == (p_add, period), (
            f"{letter}: expected (p_add={p_add}, period={period}), got {_P[letter]!r}")


def test_s2_existing_P_rows_unchanged():
    """既有 P_base / P_hotspot 两行 (p_add, period) 不变."""
    from des.registry import _P
    assert _P["P_base"] == (0.0, 1)
    assert _P["P_hotspot"] == (0.05, 3)
```

- [ ] **Step 2: 跑测试,确认失败**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py -v -k "s2_new_P or s2_existing_P"
```

Expected: `test_s2_new_P_letters_present_in_alphabet_with_family_P` / `test_s2_new_P_letters_have_gran_residue` / `test_s2_new_P_letters_have_exact_p_add_and_period` 三条 FAIL (字母还没加);`test_s2_existing_P_rows_unchanged` PASS。

- [ ] **Step 3: 在 `src/des/registry.py` 扩展 `ALPHABET` / `GRAN` / `_P`**

把 `ALPHABET` 块(原 line 13–18)替换为(保留原 6 行,追加 10 行,**顺序**与 spec §1 表一致):

```python
ALPHABET = {
    "N0": "N",
    "F4Nr1": "F", "F4Nr4": "F",
    "P_base": "P", "P_hotspot": "P",
    "BroadSweep": "Z",
    # S2: P-pool expansion (10 new P primitives — primitive-roster P pool)
    "P_aic":            "P",
    "P_ep":             "P",
    "P_fscan":          "P",
    "P_zscan":          "P",
    "P_entropy_brake":  "P",
    "P_loopswap_lite":  "P",
    "P_neutral_sink":   "P",
    "P_slow_drift":     "P",
    "P_burst_lite":     "P",
    "P_balanced":       "P",
}
```

把 `GRAN` 块(S6 已加,位置紧跟 `ALPHABET`)扩展为:

```python
GRAN: dict[str, str] = {
    "N0":         "residue",
    "F4Nr1":      "residue",
    "F4Nr4":      "residue",
    "P_base":     "residue",
    "P_hotspot":  "residue",
    "BroadSweep": "residue",
    # S2: P-pool expansion (all residue — motif P primitives are future-spec)
    "P_aic":            "residue",
    "P_ep":             "residue",
    "P_fscan":          "residue",
    "P_zscan":          "residue",
    "P_entropy_brake":  "residue",
    "P_loopswap_lite":  "residue",
    "P_neutral_sink":   "residue",
    "P_slow_drift":     "residue",
    "P_burst_lite":     "residue",
    "P_balanced":       "residue",
}
```

把 `_P` 块(原 line 34–37)替换为:

```python
_P = {    # name -> (p_add, period); effective rate = min(p_max, μ + p_add)
    "P_base":           (0.0,  1),
    "P_hotspot":        (_DELTA, 3),
    # S2: 10 new P primitives (verbatim from primitive-roster §P pool)
    "P_aic":            (0.03, 3),
    "P_ep":             (0.04, 3),
    "P_fscan":          (0.02, 5),
    "P_zscan":          (0.02, 5),
    "P_entropy_brake":  (0.01, 7),
    "P_loopswap_lite":  (0.03, 4),
    "P_neutral_sink":   (0.02, 5),
    "P_slow_drift":     (0.0,  9),
    "P_burst_lite":     (0.07, 2),
    "P_balanced":       (0.04, 3),
}
```

(`_DELTA` 是 `registry.py` 顶部已有的 `_DELTA = 0.05`;P_hotspot 行不动。)

- [ ] **Step 4: 跑新测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py -v -k "s2_new_P or s2_existing_P"
```

Expected: 4 条全 PASS。

- [ ] **Step 5: 跑全 `tests/test_registry.py`,确认 S6 加的 `GRAN` 覆盖检查仍然绿**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py -v
```

Expected: 既有 `test_gran_covers_every_alphabet_letter`(S6 Task 1 加的)以及全部其他既有断言仍然 PASS —— `GRAN` 现在覆盖了 16 个 ALPHABET 字母。

Backtrack: 若 `test_gran_covers_every_alphabet_letter` FAIL,说明 `GRAN` 漏加了某个新 P 行。逐行对照 spec §1 表补齐。

- [ ] **Step 6: 跑全 suite,确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。**注意:这一步可能会 FAIL** —— 既有 `_spectrum_for` 把 `dominant_p="P_base"` 的归一分母从「6 字母」涨到「16 字母」,所以读 `phenotype(BB0).spectrum` 具体数值的既有断言可能漂移。**这就是 Global Constraints 里说的「default-run drift 是设计兑现」**,在 Task 7 里集中处理。本步骤如果出现这一类失败,**先不动**,记录受影响测试名,Task 7 会按 RE-RECORD 政策升级它们。其他类失败(import 错、`ALPHABET` 长度被硬编码 == 6 等)必须本步骤就修。

Backtrack 判定方式:`pytest -v` 出现 `assert <number> == <number>` 的失败且失败的测试名带 "spectrum" / "P_base" / "mutation_distribution" → 属于设计漂移,留给 Task 7。其余失败 → 本步骤修。

- [ ] **Step 7: Commit**

```bash
git add src/des/registry.py tests/test_registry.py
git commit -m "feat(s2): expand ALPHABET / GRAN / _P with 10 new P primitives

Add P_aic / P_ep / P_fscan / P_zscan / P_entropy_brake / P_loopswap_lite /
P_neutral_sink / P_slow_drift / P_burst_lite / P_balanced (family='P',
gran='residue'). _P rows verbatim from primitive-roster §P pool. Shape
table (SPECTRUM_SHAPE) and _spectrum_for body extension land in Task 2/3."
```

---

### Task 2: 加 `SPECTRUM_SHAPE` 表 + module-load 值域守门(纯数据,不动 `_spectrum_for`)

**Goal:** 把三旋钮 shape 表 `SPECTRUM_SHAPE: dict[str, tuple[float, FamilyMask, float]]` 写进 `registry.py`,**全部 12 个 P 基元都给 shape 行**(包括既有 `P_base` / `P_hotspot`,值都是默认 `(1.0, None, 0.0)`)。在表定义之后立即用 `assert` 守住 power / family_mask / flatten_mix 的合法值。这一步**不动**`_spectrum_for` —— 值还没人读,纯数据扩张,既有谱行为 0 漂移。

**Files:**
- Modify: `src/des/registry.py` — 在 `_P` 块之后、`affinity` 函数之前插入 `SPECTRUM_SHAPE` + module-load 校验。
- Test: `tests/test_spectrum_shape.py` (Create)

**Interfaces:**
- Consumes: `_P` 全 12 行(Task 1)、`ALPHABET`(Task 1)。
- Produces:
  - `SPECTRUM_SHAPE: dict[str, tuple[float, FamilyMask, float]]` —— **覆盖全部 `letter for letter in _P`** 共 12 个 key,每个 value 是 `(power, family_mask, flatten_mix)`,数值 verbatim 抄 File Structure 表。
  - 模块 import 时的 `assert` 校验失败 → 直接 fail-fast 阻止后续。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_spectrum_shape.py`:

```python
# tests/test_spectrum_shape.py
"""S2 shaped mutation spectrum: SPECTRUM_SHAPE table + _spectrum_for body extension.

Default v1 alphabet pre-S2 had 6 letters and only P_base / P_hotspot in _P;
S2 grows _P to 12 rows. SPECTRUM_SHAPE is co-extensive with _P — every P key
must have a shape row. Tests are written so that adding more P primitives
post-S2 does NOT require touching this file's data."""
from __future__ import annotations
import pytest
from des import registry


# --- Task 2 surface: SPECTRUM_SHAPE table -----------------------------------

_S2_EXPECTED_SHAPE = {
    "P_base":           (1.0, None,       0.0),
    "P_hotspot":        (1.0, None,       0.0),
    "P_aic":            (2.0, None,       0.0),
    "P_ep":             (1.0, None,       0.5),
    "P_fscan":          (1.0, "F",        0.0),
    "P_zscan":          (1.0, "Z",        0.0),
    "P_entropy_brake":  (3.0, None,       0.0),
    "P_loopswap_lite":  (1.0, "adjacent", 0.0),
    "P_neutral_sink":   (1.0, "N",        0.0),
    "P_slow_drift":     (1.0, None,       0.0),
    "P_burst_lite":     (1.0, None,       0.0),
    "P_balanced":       (1.0, None,       0.0),
}


def test_spectrum_shape_covers_every_P_letter():
    """SPECTRUM_SHAPE 必须覆盖全部 _P 行,key 集合相等."""
    from des.registry import SPECTRUM_SHAPE, _P
    assert set(SPECTRUM_SHAPE.keys()) == set(_P.keys())


def test_spectrum_shape_values_match_roster_verbatim():
    """每条 (power, family_mask, flatten_mix) 与 spec §1 表一致."""
    from des.registry import SPECTRUM_SHAPE
    for letter, expected in _S2_EXPECTED_SHAPE.items():
        assert SPECTRUM_SHAPE[letter] == expected, (
            f"{letter}: expected {expected!r}, got {SPECTRUM_SHAPE[letter]!r}")


def test_spectrum_shape_power_in_legal_set():
    """power ∈ {1.0, 2.0, 3.0} —— spec §3 三旋钮锁死值域."""
    from des.registry import SPECTRUM_SHAPE
    for letter, (power, _, _) in SPECTRUM_SHAPE.items():
        assert power in (1.0, 2.0, 3.0), f"{letter}: bad power {power!r}"


def test_spectrum_shape_family_mask_in_legal_set():
    """family_mask ∈ {None, 'F', 'Z', 'N', 'adjacent'} —— spec §3."""
    from des.registry import SPECTRUM_SHAPE
    for letter, (_, mask, _) in SPECTRUM_SHAPE.items():
        assert mask in (None, "F", "Z", "N", "adjacent"), (
            f"{letter}: bad family_mask {mask!r}")


def test_spectrum_shape_flatten_mix_in_unit_interval():
    """flatten_mix ∈ [0.0, 1.0] —— spec §5 module-load assert."""
    from des.registry import SPECTRUM_SHAPE
    for letter, (_, _, mix) in SPECTRUM_SHAPE.items():
        assert 0.0 <= mix <= 1.0, f"{letter}: flatten_mix {mix!r} outside [0,1]"
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_spectrum_shape.py -v
```

Expected: 5 条全 FAIL with `ImportError: cannot import name 'SPECTRUM_SHAPE' from 'des.registry'`。

- [ ] **Step 3: 加 `SPECTRUM_SHAPE` + module-load 校验到 `src/des/registry.py`**

在 `_P = {...}` 块之后、`def affinity` 之前插入:

```python
# Mutation spectrum shape per P primitive (S2 §3). Three knobs cover all 12 P
# rows — no per-primitive special path. (power, family_mask, flatten_mix):
#   power       : 1=aff, 2=sharpen (P_aic), 3=super-sharpen (P_entropy_brake)
#   family_mask : None=all, "F"|"Z"|"N"=single family,
#                 "adjacent"=|Δrank|=1 (P_loopswap_lite)
#   flatten_mix : 0.0 default, 0.5 (P_ep — ½·aff + ½·1/(|A|-1))
# SPECTRUM_SHAPE is co-extensive with _P (every P letter has a shape row);
# adding a new P primitive REQUIRES adding both a _P row and a SPECTRUM_SHAPE row.
SPECTRUM_SHAPE: dict[str, tuple[float, "str | None", float]] = {
    "P_base":           (1.0, None,       0.0),
    "P_hotspot":        (1.0, None,       0.0),
    "P_aic":            (2.0, None,       0.0),
    "P_ep":             (1.0, None,       0.5),
    "P_fscan":          (1.0, "F",        0.0),
    "P_zscan":          (1.0, "Z",        0.0),
    "P_entropy_brake":  (3.0, None,       0.0),
    "P_loopswap_lite":  (1.0, "adjacent", 0.0),
    "P_neutral_sink":   (1.0, "N",        0.0),
    "P_slow_drift":     (1.0, None,       0.0),
    "P_burst_lite":     (1.0, None,       0.0),
    "P_balanced":       (1.0, None,       0.0),
}

# Module-load value-domain assertions (spec §5). Halt fail-fast at import time
# rather than mid-tick if a future spec adds a malformed shape row.
assert set(SPECTRUM_SHAPE.keys()) == set(_P.keys()), (
    "SPECTRUM_SHAPE must be co-extensive with _P; "
    f"missing={set(_P.keys()) - set(SPECTRUM_SHAPE.keys())}, "
    f"extra={set(SPECTRUM_SHAPE.keys()) - set(_P.keys())}")
for _letter, (_power, _mask, _mix) in SPECTRUM_SHAPE.items():
    assert _power in (1.0, 2.0, 3.0), \
        f"SPECTRUM_SHAPE[{_letter!r}].power = {_power!r} not in {{1,2,3}}"
    assert _mask in (None, "F", "Z", "N", "adjacent"), \
        f"SPECTRUM_SHAPE[{_letter!r}].family_mask = {_mask!r} not in {{None,F,Z,N,adjacent}}"
    assert 0.0 <= _mix <= 1.0, \
        f"SPECTRUM_SHAPE[{_letter!r}].flatten_mix = {_mix!r} outside [0,1]"
del _letter, _power, _mask, _mix
```

- [ ] **Step 4: 跑测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_spectrum_shape.py -v
```

Expected: 5 条全 PASS。

- [ ] **Step 5: 跑全 suite,确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿(`SPECTRUM_SHAPE` 添加是纯数据,无消费者读它,既有谱行为 0 漂移)。如果 Task 1 Step 6 残留了 spectrum-数值漂移类失败,**仍然挂在那里**(不要在本步骤 fix,留给 Task 7 集中处理)。

Backtrack: 若 module-load `assert` 失败导致大量 import 错,把 `_P` 与 `SPECTRUM_SHAPE` 的 key 集合手工 diff —— 必有一边漏行或多行。

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_spectrum_shape.py
git commit -m "feat(s2): add SPECTRUM_SHAPE table + import-time value-domain assert

12 rows (one per _P key) of (power, family_mask, flatten_mix). power ∈ {1,2,3};
family_mask ∈ {None, 'F', 'Z', 'N', 'adjacent'}; flatten_mix ∈ [0,1]. Module-
load asserts halt fail-fast on malformed shape rows. _spectrum_for body
extension lands in Task 3 — this commit changes no spectrum behavior."
```

---

### Task 3: 把三旋钮 shape 叠到 `_spectrum_for(letter)` 里

**Goal:** 扩展 `src/des/registry.py` 里的 `_spectrum_for(letter)` —— 在 S6 已经叠好的 gran-match + 等长预过滤之后,**再**叠 `(power, family_mask, flatten_mix)` 三旋钮:`w(t) = aff(src,fam(t))**power · 𝟙[mask(t)]`,然后对 P_ep 再做 `w(t) = (1-mix)·w + mix·1/(|A|-1)`,最后归一。**signature 不变**(`_spectrum_for(letter: str) -> tuple[tuple[str, float], ...]`)、**Phenotype.spectrum cache 不动**、**reproduction kernel 调用点不动**。空谱守门(`tot==0 → return ()`)继续生效。

**Files:**
- Modify: `src/des/registry.py:45-53` (`_spectrum_for` body) —— S6 落地后这块函数体已经从「只读 affinity」变成「affinity + gran-match + 等长预过滤」;本步骤在 gran-match 后、归一前再叠一层 shape。
- Test: `tests/test_spectrum_shape.py` (append)

**Interfaces:**
- Consumes: `SPECTRUM_SHAPE`(Task 2)、`affinity` / `ALPHABET` / `GRAN` / `MOTIF_LEN` / `FAMILY_RANK`(既有)。
- Produces: 同 signature 的 `_spectrum_for(letter)`,出口仍是 `(target, q)` 序列;新口径下:
  - `power`: 把 `aff` 的指数从 1 抬到 2 (P_aic) 或 3 (P_entropy_brake);**P_aic 比 P_base 更锐**(同家族 mass 占比更高)、**P_entropy_brake 比 P_aic 更锐**。
  - `family_mask`: 单家族 mask `{"F"}` / `{"Z"}` / `{"N"}` 把所有非该家族 target 的 weight 归零;`"adjacent"` mask 把所有 `|FAMILY_RANK[fam(src)] - FAMILY_RANK[fam(t)]| != 1` 的 target 归零。
  - `flatten_mix`: 仅 P_ep 取 0.5,公式 `w = (1-0.5)·aff^1 + 0.5·1/(|A|-1)` —— `|A|` 是 **live registry 全 ALPHABET 字数**,即 Task 1 后的 16(spec §2 红线)。
  - 空谱守门:`tot==0 → ()`(spec §5)。

- [ ] **Step 1: 写失败测试 — 每条 shape 的 bias 验证 + 空谱**

追加到 `tests/test_spectrum_shape.py`:

```python
# --- Task 3 surface: _spectrum_for body ------------------------------------

def _spectrum_dict(letter):
    """Helper: 把 _spectrum_for 的 (target, q) 序列转 dict 方便对照."""
    from des.registry import _spectrum_for
    return dict(_spectrum_for(letter))


def test_p_aic_is_sharper_than_p_base():
    """power=2 锐化:对 P_aic 的同家族 target,在归一后 mass 严格大于 P_base
    (因为 aff 是 [0,1] 单调,平方放大相对差异 → 同家族占比更高)."""
    base = _spectrum_dict("P_base")
    aic  = _spectrum_dict("P_aic")
    # P 家族对 P 家族 (rank-distance 0) 是最大 aff,sharpen 抬其占比
    same_family_targets = [t for t in base if registry.ALPHABET[t] == "P"]
    assert same_family_targets, "test fixture broken: no P-family target survived"
    base_mass = sum(base[t] for t in same_family_targets)
    aic_mass  = sum(aic[t]  for t in same_family_targets if t in aic)
    assert aic_mass > base_mass + 1e-9, (
        f"P_aic mass on P-family ({aic_mass:.4f}) must exceed P_base ({base_mass:.4f})")


def test_p_entropy_brake_is_sharper_than_p_aic():
    """power=3 比 power=2 更锐."""
    aic   = _spectrum_dict("P_aic")
    brake = _spectrum_dict("P_entropy_brake")
    same_family_targets = [t for t in aic if registry.ALPHABET[t] == "P"]
    aic_mass   = sum(aic[t]   for t in same_family_targets)
    brake_mass = sum(brake[t] for t in same_family_targets if t in brake)
    assert brake_mass > aic_mass + 1e-9, (
        f"P_entropy_brake mass ({brake_mass:.4f}) must exceed P_aic ({aic_mass:.4f})")


def test_p_ep_is_flatter_than_p_base_on_dominant_target():
    """flatten_mix=0.5 使 P_ep 的 mass 比 P_base 更均匀:同家族占比 < P_base."""
    base = _spectrum_dict("P_base")
    ep   = _spectrum_dict("P_ep")
    same_family_targets = [t for t in base if registry.ALPHABET[t] == "P"]
    base_mass = sum(base[t] for t in same_family_targets)
    ep_mass   = sum(ep[t]   for t in same_family_targets if t in ep)
    assert ep_mass < base_mass - 1e-9, (
        f"P_ep mass on P-family ({ep_mass:.4f}) must be < P_base ({base_mass:.4f})")


def test_p_fscan_mass_only_on_F_targets():
    """family_mask='F': 全部 mass 落在 family=='F' 的 target 上."""
    spec = _spectrum_dict("P_fscan")
    f_targets = [t for t in spec if registry.ALPHABET[t] == "F"]
    non_f_targets = [t for t in spec if registry.ALPHABET[t] != "F"]
    f_mass = sum(spec[t] for t in f_targets)
    non_f_mass = sum(spec[t] for t in non_f_targets)
    assert f_targets, "P_fscan should keep at least one F target in 16-letter v1 alphabet"
    assert non_f_mass == 0.0, f"P_fscan leaked mass to non-F targets: {non_f_mass}"
    assert abs(f_mass - 1.0) < 1e-9, f"P_fscan F-mass {f_mass} not normalized"


def test_p_zscan_mass_only_on_Z_targets():
    spec = _spectrum_dict("P_zscan")
    z_targets = [t for t in spec if registry.ALPHABET[t] == "Z"]
    non_z = sum(spec[t] for t in spec if registry.ALPHABET[t] != "Z")
    assert non_z == 0.0
    if z_targets:
        assert abs(sum(spec[t] for t in z_targets) - 1.0) < 1e-9


def test_p_neutral_sink_mass_only_on_N_targets():
    spec = _spectrum_dict("P_neutral_sink")
    n_targets = [t for t in spec if registry.ALPHABET[t] == "N"]
    non_n = sum(spec[t] for t in spec if registry.ALPHABET[t] != "N")
    assert non_n == 0.0
    if n_targets:
        assert abs(sum(spec[t] for t in n_targets) - 1.0) < 1e-9


def test_p_loopswap_lite_mass_only_on_adjacent_rank():
    """family_mask='adjacent' (|Δrank|=1): mass 仅落在与 P 家族 rank 相邻
    (即 |FAMILY_RANK['P'] - FAMILY_RANK[fam(t)]|==1) 的 target."""
    from des.types import FAMILY_RANK
    src_rank = FAMILY_RANK["P"]
    spec = _spectrum_dict("P_loopswap_lite")
    bad = [t for t, q in spec.items()
           if q > 0 and abs(FAMILY_RANK[registry.ALPHABET[t]] - src_rank) != 1]
    assert not bad, f"P_loopswap_lite leaked mass to non-adjacent targets: {bad}"


def test_burst_lite_slow_drift_balanced_share_p_base_shape():
    """三个 'shape ≡ P_base' 的 P 行(P_burst_lite / P_slow_drift / P_balanced)
    的 spectrum 应与 P_base 字节级一致 —— 它们只在 (p_add, period) 上不同."""
    base = _spectrum_dict("P_base")
    for letter in ("P_burst_lite", "P_slow_drift", "P_balanced"):
        assert _spectrum_dict(letter) == base, (
            f"{letter} spectrum diverges from P_base shape")


def test_spectrum_normalizes_to_unit_sum_for_every_P_letter():
    """每个 P 行的 spectrum 都归一到 Σq=1(或为空 ())."""
    from des.registry import _spectrum_for, _P
    for letter in _P:
        spec = _spectrum_for(letter)
        if spec == ():
            continue
        total = sum(q for _, q in spec)
        assert abs(total - 1.0) < 1e-9, f"{letter}: Σq={total} != 1"


def test_p_fscan_returns_empty_when_no_F_letter_in_alphabet(monkeypatch):
    """family_mask='F' + alphabet 里没有 F 字母 → tot=0 → 返回 () (spec §5)."""
    # 把 ALPHABET 暂时缩成「只 P + N」(去掉 F4Nr1 / F4Nr4 / BroadSweep)
    minimal = {k: v for k, v in registry.ALPHABET.items() if v not in ("F", "Z")}
    monkeypatch.setattr(registry, "ALPHABET", minimal)
    # GRAN 同步缩(_spectrum_for 用 GRAN[t] 做 gran-match)
    minimal_gran = {k: v for k, v in registry.GRAN.items() if k in minimal}
    monkeypatch.setattr(registry, "GRAN", minimal_gran)
    from des.registry import _spectrum_for
    assert _spectrum_for("P_fscan") == ()


def test_p_ep_flatten_uses_full_alphabet_size_in_denominator():
    """spec §2 红线:flatten 项 1/(|A|-1) 的 |A| 用 full ALPHABET 大小,
    不做子集排除. 16 字母 alphabet → 分母是 15."""
    from des.registry import _spectrum_for, ALPHABET, SPECTRUM_SHAPE, affinity
    A = len(ALPHABET)
    src_fam = ALPHABET["P_ep"]
    power, mask, mix = SPECTRUM_SHAPE["P_ep"]
    # 重算预期 spectrum (gran-match: P_ep 是 residue → 全部 residue target 都过滤通过)
    raw = {}
    for t in ALPHABET:
        if t == "P_ep":
            continue
        if registry.GRAN[t] != "residue":
            continue
        w = affinity(src_fam, ALPHABET[t]) ** power
        # mask=None → 不过滤 family
        w = (1 - mix) * w + mix * (1.0 / (A - 1))
        raw[t] = w
    tot = sum(raw.values())
    expected = {t: raw[t] / tot for t in raw}
    actual = dict(_spectrum_for("P_ep"))
    assert set(actual) == set(expected)
    for t in expected:
        assert abs(actual[t] - expected[t]) < 1e-9, (
            f"P_ep target {t}: expected {expected[t]:.6f}, got {actual[t]:.6f}")
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_spectrum_shape.py -v
```

Expected: Task 3 这批新测试全 FAIL —— 现行 `_spectrum_for` 只读 `affinity`,不读 `SPECTRUM_SHAPE`,所以 `P_aic` / `P_ep` / `P_fscan` / `P_loopswap_lite` 等的 spectrum 与 `P_base` 完全相同(失去 bias)。Task 2 加的 5 条数据/值域守门测试仍 PASS。

- [ ] **Step 3: 重写 `_spectrum_for(letter)` 函数体**

把 `src/des/registry.py:45-53` 的 `_spectrum_for` 替换为(**保留 S6 的 gran-match + 等长预过滤,叠加三旋钮**):

```python
def _spectrum_for(letter: str) -> tuple[tuple[str, float], ...]:
    """Family-distance spectrum, gran-matched + equal-length pre-filtered (S6),
    shape-modulated by SPECTRUM_SHAPE (S2). Pure function of the alphabet.

    Three-knob shape from SPECTRUM_SHAPE[letter] = (power, family_mask, flatten_mix):
      w(t) = aff(fam(letter), fam(t)) ** power · 𝟙[mask(t)]
      w(t) = (1 - mix) · w(t) + mix · 1 / (|A| - 1)            # only when mix > 0
    Targets must additionally pass S6 gran-match + equal-length pre-filter.
    Renormalized to Σq=1 across surviving targets; empty pre-filter → ()."""
    src_fam = ALPHABET[letter]
    src_gran = GRAN[letter]
    src_len = MOTIF_LEN.get(letter)
    # SPECTRUM_SHAPE may be missing for non-P letters historically; default-safe.
    power, mask, mix = SPECTRUM_SHAPE.get(letter, (1.0, None, 0.0))
    src_rank = FAMILY_RANK[src_fam]
    A = len(ALPHABET)

    survivors: dict[str, float] = {}
    for t in ALPHABET:
        if t == letter:
            continue
        # S6 predicates: gran-match + equal-length (motif only)
        if GRAN[t] != src_gran:
            continue
        if src_gran == "motif" and MOTIF_LEN[t] != src_len:
            continue
        # S2 family_mask predicate
        if mask is None:
            pass
        elif mask == "adjacent":
            if abs(FAMILY_RANK[ALPHABET[t]] - src_rank) != 1:
                continue
        else:
            # mask is a single family letter ("F" / "Z" / "N")
            if ALPHABET[t] != mask:
                continue
        # S2 power
        w = affinity(src_fam, ALPHABET[t]) ** power
        # S2 flatten_mix (only meaningful when mix > 0; uniform 1/(|A|-1))
        if mix > 0.0:
            w = (1.0 - mix) * w + mix * (1.0 / (A - 1))
        survivors[t] = w

    tot = sum(survivors.values())
    if tot == 0.0:
        return ()
    return tuple((t, w / tot) for t, w in sorted(survivors.items()))
```

注意:`FAMILY_RANK` 已经在 `registry.py` 顶部 `from des.types import ... FAMILY_RANK` 进口,`affinity` 是同文件本地函数,`GRAN` / `MOTIF_LEN` / `SPECTRUM_SHAPE` 由 Task 1/2 / S6 加进 registry,**不需要新 import**。

- [ ] **Step 4: 跑测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_spectrum_shape.py -v
```

Expected: Task 2 的 5 条 + Task 3 的 11 条全 PASS。

Backtrack:
- 若 `test_p_aic_is_sharper_than_p_base` FAIL,可能是 `power` 没生效 —— 检查 `**` 优先级、确认 `w = affinity(...) ** power` 而非 `affinity(...) ** 1`。
- 若 `test_p_loopswap_lite_mass_only_on_adjacent_rank` FAIL,检查 `"adjacent"` 分支用了 `FAMILY_RANK[ALPHABET[t]]`(通过 letter 取 family,再取 rank),不是直接 `FAMILY_RANK[t]`。
- 若 `test_p_ep_flatten_uses_full_alphabet_size_in_denominator` FAIL,检查 `A = len(ALPHABET)` 是 16 而非 15(spec §2:**full** ALPHABET,不做 -1)。

- [ ] **Step 5: 跑全 suite,记录受影响的 spectrum-数值断言**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q 2>&1 | tee s2_task3_full_suite.log
```

Expected: 大概率会出现 spectrum-数值类的回归失败(`tests/test_registry.py::test_dominant_p_*` / `tests/test_motif.py::test_spectrum_residue_path_byte_identical_to_legacy` / `tests/test_phenotype_cache.py` 里读 `Phenotype.spectrum` 具体数值的断言)。**这是设计漂移,不是 bug**(Global Constraints + spec §6)。

操作:把日志里所有以 `FAILED tests/...` 开头、且失败原因是「数值不等」的测试名记录到 `s2_task3_affected_tests.txt`(可手抄、可 `grep '^FAILED' s2_task3_full_suite.log > s2_task3_affected_tests.txt`)。**Task 7 集中处理这份名单**;本步骤不动这些测试。

非数值类失败(import 错、属性错、shape 校验崩)必须本步骤就修;先 root-cause,常见原因:
- `SPECTRUM_SHAPE.get(letter, (1.0, None, 0.0))` 默认值与 module-load 校验冲突 → 把默认值改回必须存在,即不要 `.get` 默认,改为 `SPECTRUM_SHAPE[letter]`(因为 `_spectrum_for` 的 caller 总是 `dominant_p`,而 `dominant_p` 只能取 `_P` 里的 key,且 `SPECTRUM_SHAPE` Task 2 已守门 `keys ⊇ _P.keys`)。
- F4Nr1 / F4Nr4 也会被 `phenotype()` 当作 dominant_p 候选?——不会,`phenotype()` 的 dominant_p 检索只在 `letter in _P` 分支里赋值,F 字母走不到。

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_spectrum_shape.py
git commit -m "feat(s2): _spectrum_for reads SPECTRUM_SHAPE three-knob table

power (sharpen) / family_mask (single-family or adjacent-rank) / flatten_mix
(P_ep) layered after S6's gran-match + equal-length pre-filter. Signature
unchanged; Phenotype.spectrum cache + reproduction kernel call site
unchanged. P_burst_lite / P_slow_drift / P_balanced share P_base shape
(differ only in p_add/period). Default-run distribution shifts because
the |A| denominator grew 6→16 — designed re-baseline; affected fixtures
are re-recorded in Task 7."
```

---

<!-- SECTION:TASK_4 -->

---

<!-- SECTION:TASK_5 -->

---

<!-- SECTION:TASK_6 -->

---

<!-- SECTION:TASK_7 -->

---

## Self-Review

<!-- SECTION:SELF_REVIEW -->

---

## Execution Handoff

<!-- SECTION:EXECUTION_HANDOFF -->
