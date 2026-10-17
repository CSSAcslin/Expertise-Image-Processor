"""Exact finite-value quantiles, including bounded external GPU rank selection."""
import numpy as np


def finite_quantile(values, q, axes, xp=np):
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    kept = [i for i in range(values.ndim) if i not in axes]
    shape = tuple(values.shape[i] for i in kept)
    matrix = values.transpose(kept + list(axes)).reshape((*shape, -1))
    valid = xp.isfinite(matrix)
    count = valid.sum(axis=-1)
    ordered = xp.sort(xp.where(valid, matrix, xp.inf), axis=-1)
    position = xp.maximum(count-1, 0)*q
    lower, upper = xp.floor(position).astype(xp.int64), xp.ceil(position).astype(xp.int64)
    a = xp.take_along_axis(ordered, lower[..., None], axis=-1)[..., 0]
    b = xp.take_along_axis(ordered, upper[..., None], axis=-1)[..., 0]
    fraction = position-lower
    with np.errstate(invalid="ignore"):
        return xp.where(count, (1-fraction)*a+fraction*b, xp.nan)


def external_quantile(values, n, q, xp, host, budget, check):
    """Select exact float64 ranks; large rows use eight bounded radix passes.

    Invalid spool values are +Inf. Rank is based only on the finite count n.
    Float-to-ordered-uint mapping preserves every finite float64 value, including
    negative values and subnormals; no histogram approximation is introduced.
    """
    if not n:
        return np.nan
    position = (int(n)-1)*q
    ranks = (int(np.floor(position)), int(np.ceil(position)))
    if len(values)*8*8 <= budget:
        ordered = xp.partition(xp.asarray(values), sorted(set(ranks)))
        pair = host(ordered[xp.asarray(ranks, dtype=xp.int64)])
    else:
        block = max(1, (budget-4096)//96)
        states = {rank: (0, rank) for rank in set(ranks)}
        sign = 1 << 63
        for shift in range(56, -1, -8):
            check()
            prefixes = {prefix for prefix, _ in states.values()}
            histograms = {prefix: xp.zeros(256, dtype=xp.int64) for prefix in prefixes}
            mask = ((1 << 64)-1) ^ ((1 << (shift+8))-1)
            for start in range(0, len(values), block):
                check()
                chunk = xp.asarray(values[start:start+block], dtype=xp.float64)
                bits = chunk.view(xp.uint64)
                keys = xp.where((bits & xp.uint64(sign)) != 0, ~bits, bits ^ xp.uint64(sign))
                for prefix, hist in histograms.items():
                    selected = keys[xp.isfinite(chunk) & ((keys & xp.uint64(mask)) == xp.uint64(prefix))]
                    digits = ((selected >> xp.uint64(shift)) & xp.uint64(255)).astype(xp.int64)
                    hist += xp.bincount(digits, minlength=256).astype(xp.int64)
            for rank, (prefix, remaining) in list(states.items()):
                cumulative = np.cumsum(host(histograms[prefix]), dtype=np.int64)
                digit = int(np.searchsorted(cumulative, remaining, side="right"))
                if digit > 255:
                    raise ValueError("精确分位值有限计数与数据不一致")
                remaining -= int(cumulative[digit-1]) if digit else 0
                states[rank] = (prefix | (digit << shift), remaining)
        pair = []
        for rank in ranks:
            key = states[rank][0]
            bits = key ^ sign if key & sign else (~key & ((1 << 64)-1))
            pair.append(np.array([bits], dtype=np.uint64).view(np.float64)[0])
    fraction = position-ranks[0]
    return (1-fraction)*pair[0]+fraction*pair[1]
