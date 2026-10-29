"""Execute real float64 CUDA and EIP fused kernels with the selected Python.

Exit nonzero on missing CUDA, compilation errors, or numerical disagreement.
No application settings or experimental data are changed.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check():
    from eip.modes.split_correlation.backend import Backend
    from eip.modes.split_correlation.correlation import RollingMoments, direct_g2
    import cupy as cp

    backend = Backend("gpu", 64 * 1024**2)
    try:
        if cp.sum(cp.arange(8, dtype=cp.float64)**2).item() != 140.0:
            raise AssertionError("CUDA float64 self-test failed")
        pairs = np.random.default_rng(2710).uniform(2, 12, (23, 2, 3, 4))
        pairs[6, 0, 1, 1] = np.nan
        pairs[9, 1, 2, 2] = np.inf
        uploaded = cp.asarray(pairs)
        errors = []
        for lag in (-2, 0, 3):
            moments = RollingMoments(pairs[0], SimpleNamespace(min_mean=1e-9, min_samples=2), backend)
            # Interleaved channels exercise non-contiguous raw kernel inputs.
            moments.add_many(uploaded[3:19, 0], uploaded[3+lag:19+lag, 1])
            actual, count = moments.result()
            expected, expected_count = direct_g2(pairs[3:19, 0], pairs[3+lag:19+lag, 1])
            np.testing.assert_array_equal(count, expected_count)
            np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14, equal_nan=True)
            assert moments.implementation == "cupy_fused_float64"
            errors.append(dict(lag=lag, max_absolute_error=float(np.nanmax(np.abs(actual-expected)))))
        cp.cuda.Stream.null.synchronize()
        config = io.StringIO()
        with redirect_stdout(config):
            cp.show_config()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        free, total = cp.cuda.runtime.memGetInfo()
        return dict(status="passed", python=sys.executable, python_version=platform.python_version(),
                    cupy=cp.__version__, cupy_path=cp.__file__, numpy=np.__version__,
                    gpu_name=name.decode() if isinstance(name, bytes) else str(name),
                    compute_capability=cp.cuda.Device(0).compute_capability,
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                    cuda_driver=cp.cuda.runtime.driverGetVersion(),
                    free_bytes=free, total_bytes=total,
                    implementation="cupy_fused_float64", signed_lag_checks=errors,
                    configuration=config.getvalue())
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON diagnostic file")
    args = parser.parse_args()
    try:
        report = check()
    except Exception as error:
        report = dict(status="failed", python=sys.executable,
                      error=f"{type(error).__name__}: {error}")
    value = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value + "\n", encoding="utf-8")
    print(value)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
