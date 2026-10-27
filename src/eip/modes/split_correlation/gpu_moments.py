"""Lazy CuPy kernels for the existing float64 fluctuation estimator.

Each pixel keeps the same sample order as ``RollingMoments.add``. There are no
parallel reductions, atomics, or lower-precision accumulators. Disabling FMA
retains the separate products and additions of the array implementation.
"""
from __future__ import annotations

from functools import lru_cache


_OPTIONS = ("--fmad=false",)

_UPDATE = r"""
    const bool valid = isfinite(left) && isfinite(right)
                    && isfinite(base_l) && isfinite(base_r);
    const double dl = valid ? left - base_l : 0.0;
    const double dr = valid ? right - base_r : 0.0;
    sum_l += sign * dl;
    sum_r += sign * dr;
    sum_lr += sign * (dl * dr);
    samples += sign * (valid ? 1.0 : 0.0);
"""

_RESULT = r"""
    const double safe = samples > 1.0 ? samples : 1.0;
    const double ml = sum_l / safe;
    const double mr = sum_r / safe;
    const double il = base_l + ml;
    const double ir = base_r + mr;
    const double covariance = sum_lr / safe - ml * mr;
    const double value = 1.0 + covariance / (il * ir);
    result = (samples >= min_samples && il > min_mean && ir > min_mean
              && isfinite(value)) ? value : nan("");
"""


@lru_cache(maxsize=1)
def moment_kernels(cp):
    """Create kernels only for the actual CuPy module; do not import it on CPU.

    ``raw`` ElementwiseKernel inputs retain CuPy's stride-aware flat indexing.
    In particular, the batched kernel does not require a contiguous input copy.
    Kernel compilation and execution errors deliberately propagate to the task.
    """
    outputs = "float64 sum_l, float64 sum_r, float64 sum_lr, float64 samples"
    add = cp.ElementwiseKernel(
        "float64 left, float64 right, float64 base_l, float64 base_r, int64 sign",
        outputs, _UPDATE, "eip_moments_add_float64", options=_OPTIONS)
    add_many = cp.ElementwiseKernel(
        "raw float64 left_batch, raw float64 right_batch, "
        "float64 base_l, float64 base_r, int64 sign, int64 batch_count",
        outputs,
        r"""
            double acc_l = sum_l;
            double acc_r = sum_r;
            double acc_lr = sum_lr;
            double acc_samples = samples;
            const bool base_valid = isfinite(base_l) && isfinite(base_r);
            for (long long sample = 0; sample < batch_count; ++sample) {
                const double left = left_batch[sample * _ind.size() + i];
                const double right = right_batch[sample * _ind.size() + i];
                const bool valid = base_valid && isfinite(left) && isfinite(right);
                const double dl = valid ? left - base_l : 0.0;
                const double dr = valid ? right - base_r : 0.0;
                acc_l += sign * dl;
                acc_r += sign * dr;
                acc_lr += sign * (dl * dr);
                acc_samples += sign * (valid ? 1.0 : 0.0);
            }
            sum_l = acc_l;
            sum_r = acc_r;
            sum_lr = acc_lr;
            samples = acc_samples;
        """,
        "eip_moments_add_many_float64", options=_OPTIONS)
    inputs = ("float64 sum_l, float64 sum_r, float64 sum_lr, float64 samples, "
              "float64 base_l, float64 base_r, float64 min_mean, int64 min_samples")
    result = cp.ElementwiseKernel(
        inputs, "float64 result, uint32 count",
        _RESULT + '\ncount = (unsigned int)rint(samples);',
        "eip_moments_result_float64", options=_OPTIONS)
    result_only = cp.ElementwiseKernel(
        inputs, "float64 result", _RESULT,
        "eip_moments_result_only_float64", options=_OPTIONS)
    return add, add_many, result, result_only
