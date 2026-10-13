"""Finite-value statistics with explicit reduced axes and bounded working arrays."""
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np


def retained_axes(axes, scope):
    if scope == "samples":
        return axes
    if scope == "all":
        return ""
    return "".join(a for a in axes if a in "PC")


def joint_values(values, paired, validity):
    if paired and validity == "joint":
        return np.where(np.isfinite(values).all(axis=0)[None, ...], values, np.nan)
    return values


def reduce_finite(a, axes, mode, q=.5):
    a = np.asarray(a, dtype=np.float64)
    valid = np.isfinite(a)
    count = valid.sum(axis=axes, dtype=np.uint64)
    if mode == "count":
        return count, count
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode in {"mean", "sum"}:
            total = np.where(valid, a, 0).sum(axis=axes)
            value = total / np.maximum(count, 1) if mode == "mean" else total
        elif mode in {"min", "max"}:
            fill = np.inf if mode == "min" else -np.inf
            value = getattr(np, mode)(np.where(valid, a, fill), axis=axes)
        elif mode in {"variance", "std"}:
            mean = np.where(valid, a, 0).sum(axis=axes, keepdims=True) / np.maximum(
                valid.sum(axis=axes, keepdims=True), 1,
            )
            delta = np.where(valid, a - mean, 0)
            value = np.square(delta).sum(axis=axes) / np.maximum(count, 1)
            if mode == "std":
                value = np.sqrt(value)
        elif mode == "quantile":
            value = np.nanquantile(np.where(valid, a, np.nan), q, axis=axes, method="linear")
        else:
            raise ValueError("未知统计量")
    return np.where(count, value, np.nan), count


def global_statistic(contract, parameters, get_sample, context, workdir):
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
    count = np.zeros(components, dtype=np.uint64)
    mode = parameters["mode"]
    total = np.zeros(components)
    mean, m2 = np.zeros(components), np.zeros(components)
    extreme = np.full(components, np.inf if mode == "min" else -np.inf)
    context.log(f"全域统计：{mode}，{contract.length} 个上游样本")

    def scan(spool=None):
        interval = max(1, contract.length // 20)
        for index in range(contract.length):
            context.check_cancelled()
            values = joint_values(get_sample(index), contract.paired, parameters["validity"])
            matrix = np.asarray(values, dtype=np.float64).transpose(order).reshape(components, width)
            finite = np.isfinite(matrix)
            batch_count = finite.sum(axis=1, dtype=np.uint64)
            combined = count + batch_count
            if spool is not None:
                spool[:, index*width:(index+1)*width] = np.where(finite, matrix, np.inf)
            elif mode in {"min", "max"}:
                fill = np.inf if mode == "min" else -np.inf
                batch = getattr(np, mode)(np.where(finite, matrix, fill), axis=1)
                np.minimum(extreme, batch, out=extreme) if mode == "min" else np.maximum(extreme, batch, out=extreme)
            elif mode in {"mean", "variance", "std", "sum"}:
                batch_sum = np.where(finite, matrix, 0).sum(axis=1)
                if mode == "sum":
                    total[:] += batch_sum
                else:
                    batch_mean = batch_sum / np.maximum(batch_count, 1)
                    delta = batch_mean - mean
                    if mode in {"variance", "std"}:
                        residual = np.where(finite, matrix - batch_mean[:, None], 0)
                        m2[:] += np.square(residual).sum(axis=1) + np.square(delta) * (
                            count.astype(float) * batch_count / np.maximum(combined, 1)
                        )
                    mean[:] += delta * (batch_count / np.maximum(combined, 1))
            count[:] = combined
            if (index+1) % interval == 0 or index+1 == contract.length:
                context.log(f"全域统计：{index+1}/{contract.length}")

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
                value = np.full(components, np.nan)
                for component, n in enumerate(count):
                    context.check_cancelled()
                    if n:
                        if component % max(1, components//20) == 0:
                            context.log(f"精确全域分位值：选择排序位置 {component+1}/{components}")
                        position = (int(n)-1) * parameters["quantile"]
                        lower, upper = int(np.floor(position)), int(np.ceil(position))
                        spool[component].partition((lower, upper))
                        fraction = position - lower
                        value[component] = (1-fraction)*spool[component, lower] + fraction*spool[component, upper]
            finally:
                spool._mmap.close()
    else:
        scan()
        value = count if mode == "count" else total if mode == "sum" else extreme if mode in {"min", "max"} else mean if mode == "mean" else m2 / np.maximum(count, 1)
        if mode == "std":
            value = np.sqrt(value)
        if mode != "count":
            value = np.where(count, value, np.nan)
    return np.asarray(value).reshape(shape), count.reshape(shape)
