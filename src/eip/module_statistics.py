"""Finite-value statistics with explicit reduced axes and bounded working arrays."""
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
from eip.exact_quantile import finite_quantile, external_quantile


def retained_axes(axes, scope):
    if scope == "samples":
        return axes
    if scope == "all":
        return ""
    return "".join(a for a in axes if a in "PC")


def joint_values(values, paired, validity, xp=np):
    if paired and validity == "joint":
        return xp.where(xp.isfinite(values).all(axis=0)[None, ...], values, xp.nan)
    return values


def reduce_finite(a, axes, mode, q=.5, xp=np):
    a = xp.asarray(a, dtype=xp.float64)
    valid = xp.isfinite(a)
    count = valid.sum(axis=axes, dtype=xp.uint64)
    if mode == "count":
        return count, count
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode in {"mean", "sum"}:
            total = xp.where(valid, a, 0).sum(axis=axes)
            value = total / xp.maximum(count, 1) if mode == "mean" else total
        elif mode in {"min", "max"}:
            fill = xp.inf if mode == "min" else -xp.inf
            value = getattr(xp, mode)(xp.where(valid, a, fill), axis=axes)
        elif mode in {"variance", "std"}:
            mean = xp.where(valid, a, 0).sum(axis=axes, keepdims=True) / xp.maximum(
                valid.sum(axis=axes, keepdims=True), 1,
            )
            delta = xp.where(valid, a - mean, 0)
            value = xp.square(delta).sum(axis=axes) / xp.maximum(count, 1)
            if mode == "std":
                value = xp.sqrt(value)
        elif mode == "quantile":
            value = np.nanquantile(np.where(valid, a, np.nan), q, axis=axes, method="linear") if xp is np else finite_quantile(a, q, axes, xp)
        else:
            raise ValueError("未知统计量")
    return xp.where(count, value, xp.nan), count


def global_statistic(contract, parameters, get_sample, context, workdir, *, xp=np, host=np.asarray, batch_size=1, device_budget=0):
    """Reduce the actual complete upstream, including edits and Q/N sequences.

    Non-quantile reductions stream frame by frame. Exact quantiles use an in-place
    disk-backed partition; neither approximate sampling nor a full RAM stack.
    """
    axes = retained_axes(contract.axes, parameters["scope"])
    shape = tuple(s for a, s in zip(contract.axes, contract.shape) if a in axes)
    order = [i for i, a in enumerate(contract.axes) if a in axes]
    order += [i for i, a in enumerate(contract.axes) if a not in axes]
    components = max(1, int(np.prod(shape)))
    width = max(1, int(np.prod(contract.shape))) // components
    count = xp.zeros(components, dtype=xp.uint64)
    mode = parameters["mode"]
    total = xp.zeros(components)
    mean, m2 = xp.zeros(components), xp.zeros(components)
    baseline = xp.zeros(components)
    extreme = xp.full(components, xp.inf if mode == "min" else -xp.inf)
    context.log(f"全域统计：{mode}，{contract.length} 个上游样本")

    def scan(spool=None):
        interval = max(1, contract.length // 20)
        for index in range(0, contract.length, batch_size):
            context.check_cancelled()
            stop = min(contract.length, index+batch_size)
            samples = [joint_values(get_sample(j), contract.paired, parameters["validity"], xp) for j in range(index, stop)]
            values = xp.stack(samples) if len(samples)>1 else samples[0][None, ...]
            batch_order = [k+1 for k in order[:len(axes)]]+[0]+[k+1 for k in order[len(axes):]]
            matrix = xp.asarray(values, dtype=xp.float64).transpose(batch_order).reshape(components, (stop-index)*width)
            finite = xp.isfinite(matrix)
            batch_count = finite.sum(axis=1, dtype=xp.uint64)
            combined = count + batch_count
            if spool is not None:
                spool[:, index*width:stop*width] = host(xp.where(finite, matrix, xp.inf))
            elif mode in {"min", "max"}:
                fill = xp.inf if mode == "min" else -xp.inf
                batch = getattr(xp, mode)(xp.where(finite, matrix, fill), axis=1)
                xp.minimum(extreme, batch, out=extreme) if mode == "min" else xp.maximum(extreme, batch, out=extreme)
            elif mode in {"mean", "variance", "std", "sum"}:
                if mode == "sum":
                    total[:] += xp.where(finite, matrix, 0).sum(axis=1)
                else:
                    first_valid=xp.argmax(finite,axis=1)
                    candidate=xp.take_along_axis(matrix,first_valid[:,None],axis=1)[:,0]
                    baseline[:]=xp.where((count==0)&(batch_count>0),candidate,baseline)
                    shifted=xp.where(finite,matrix-baseline[:,None],0)
                    batch_sum=shifted.sum(axis=1)
                    batch_mean = batch_sum / xp.maximum(batch_count, 1)
                    delta = batch_mean - mean
                    if mode in {"variance", "std"}:
                        residual = xp.where(finite, shifted - batch_mean[:, None], 0)
                        m2[:] += xp.square(residual).sum(axis=1) + xp.square(delta) * (
                            count.astype(float) * batch_count / xp.maximum(combined, 1)
                        )
                    mean[:] += delta * (batch_count / xp.maximum(combined, 1))
            count[:] = combined
            if index//interval != stop//interval or stop == contract.length:
                context.log(f"全域统计：{stop}/{contract.length}")

    if mode == "quantile":
        size = components * width * contract.length * 8
        directory = Path(workdir)
        directory.mkdir(parents=True, exist_ok=True)
        if size + 32*1024**2 > shutil.disk_usage(directory).free:
            raise ValueError("精确全域分位值的临时磁盘空间不足；请裁剪范围或更换输出盘")
        with tempfile.TemporaryDirectory(prefix="quantile-", dir=directory) as folder:
            spool = np.lib.format.open_memmap(Path(folder) / "values.npy", mode="w+", dtype="float64",
                                             shape=(components, width*contract.length))
            try:
                scan(spool)
                value = xp.full(components, xp.nan)
                quantile_block = max(0, device_budget//max(1, width*contract.length*8*8)) if device_budget else 0
                if device_budget and quantile_block:
                    for start in range(0, components, quantile_block):
                        context.check_cancelled()
                        value[start:start+quantile_block] = finite_quantile(xp.asarray(spool[start:start+quantile_block]), parameters["quantile"], (1,), xp)
                for component, n in enumerate(host(count)):
                    context.check_cancelled()
                    if device_budget and quantile_block:
                        break
                    if n:
                        if component % max(1, components//20) == 0:
                            context.log(f"精确全域分位值：选择排序位置 {component+1}/{components}")
                        if device_budget:
                            value[component] = external_quantile(spool[component], int(n), parameters["quantile"], xp, host, device_budget, context.check_cancelled)
                            continue
                        position = (int(n)-1) * parameters["quantile"]
                        lower, upper = int(np.floor(position)), int(np.ceil(position))
                        spool[component].partition((lower, upper))
                        fraction = position - lower
                        value[component] = (1-fraction)*spool[component, lower] + fraction*spool[component, upper]
            finally:
                spool._mmap.close()
    else:
        scan()
        value = count if mode == "count" else total if mode == "sum" else extreme if mode in {"min", "max"} else baseline+mean if mode == "mean" else m2 / xp.maximum(count, 1)
        if mode == "std":
            value = xp.sqrt(value)
        if mode != "count":
            value = xp.where(count, value, xp.nan)
    return xp.asarray(value).reshape(shape), count.reshape(shape)
