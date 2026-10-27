"""Numerical contract for fused g² updates; CUDA checks require real hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

from eip.modes.split_correlation.correlation import RollingMoments, direct_g2


class HostBackend:
    def __init__(self, name="cpu"):
        self.xp,self.name = np,name

    def host(self, value):
        return np.asarray(value)


def moments(baseline, backend=None, **config):
    return RollingMoments(baseline,SimpleNamespace(min_mean=config.get("min_mean",1e-9),
                                                 min_samples=config.get("min_samples",2)),
                          backend or HostBackend())


@pytest.mark.parametrize("shape",[(),(7,),(3,4)])
@pytest.mark.parametrize("advertised_name",["cpu","gpu"])
def test_host_batch_retains_sequential_float64_updates(shape,advertised_name):
    rng = np.random.default_rng(91)
    baseline = rng.uniform(1e6,2e6,(2,*shape))
    left = baseline[0]+rng.normal(0,.01,(19,*shape))
    right = baseline[1]+rng.normal(0,.01,(19,*shape))
    left[3] = np.nan
    right[7] = np.inf
    backend = HostBackend(advertised_name)
    batch,scalar = moments(baseline,backend),moments(baseline,backend)
    batch.add_many(left[:11],right[:11])
    batch.add_many(left[11:],right[11:])
    batch.add_many(left[2:5],right[2:5],-1)
    for l,r in zip(left,right):
        scalar.add(l,r)
    for l,r in zip(left[2:5],right[2:5]):
        scalar.add(l,r,-1)
    assert not batch.fused_gpu and batch.implementation == "array_float64"
    assert batch._kernels is None
    np.testing.assert_array_equal(batch.sums,scalar.sums)
    actual,count = batch.result()
    expected,n = scalar.result()
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(count,n)
    without,nothing = batch.result(count=False)
    np.testing.assert_array_equal(without,actual)
    assert nothing is None
    without,nothing = batch.result(device=True,count=False)
    np.testing.assert_array_equal(without,actual)
    assert nothing is None


def test_batch_shape_validation_and_empty_batch_do_not_modify_sums():
    m = moments(np.ones((2,3,4)))
    for left,right in [(np.ones((5,3,4)),np.ones((4,3,4))),
                       (np.ones((5,3,1)),np.ones((5,3,1))),
                       (np.ones((3,4)),np.ones((3,4)))]:
        with pytest.raises(ValueError,match="批量配对"):
            m.add_many(left,right)
    m.add_many(np.empty((0,3,4)),np.empty((0,3,4)))
    np.testing.assert_array_equal(m.sums,np.zeros((4,3,4)))


@pytest.fixture(scope="module")
def cuda_backend():
    from eip.modes.split_correlation.backend import Backend,devices
    if not devices()["gpu"]:
        pytest.skip("需要真实 CuPy / CUDA 设备")
    b = Backend("gpu",64*1024**2)
    try:
        yield b
    finally:
        b.close()


@pytest.mark.parametrize("shape",[(),(9,),(3,4)])
@pytest.mark.parametrize("lag",[-2,0,3])
def test_cuda_fused_updates_match_cpu_at_signed_lags(cuda_backend,shape,lag):
    rng = np.random.default_rng(82)
    pairs = rng.uniform(2,12,(23,2,*shape))
    pairs[6,0] = np.nan
    pairs[9,1] = np.inf
    baseline = np.array(pairs[0],copy=True)
    cpu,gpu = moments(baseline),moments(baseline,cuda_backend)
    cp = cuda_backend.xp
    uploaded = cp.asarray(pairs)
    for j in range(3,19):
        cpu.add(pairs[j,0],pairs[j+lag,1])
        gpu.add(uploaded[j,0],uploaded[j+lag,1])
    assert gpu.fused_gpu and gpu.implementation == "cupy_fused_float64"
    np.testing.assert_allclose(cp.asnumpy(gpu.sums),cpu.sums,rtol=0,atol=0)
    actual,count = gpu.result()
    expected,n = cpu.result()
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=2e-15,equal_nan=True)
    np.testing.assert_array_equal(count,n)
    reference,reference_count = direct_g2(pairs[3:19,0],pairs[3+lag:19+lag,1])
    np.testing.assert_allclose(actual,reference,rtol=2e-14,atol=2e-14,equal_nan=True)
    np.testing.assert_array_equal(count,reference_count)
    result,nothing = gpu.result(device=True,count=False)
    assert isinstance(result,cp.ndarray) and nothing is None
    np.testing.assert_array_equal(cp.asnumpy(result),actual)
    # Sliding-window removal must use the same joint validity as insertion.
    cpu.add_many(pairs[3:8,0],pairs[3+lag:8+lag,1],-1)
    gpu.add_many(uploaded[3:8,0],uploaded[3+lag:8+lag,1],-1)
    np.testing.assert_array_equal(cp.asnumpy(gpu.sums),cpu.sums)


@pytest.mark.parametrize("shape",[(),(7,),(3,4)])
def test_cuda_batch_matches_ordered_updates_and_accepts_strided_views(cuda_backend,shape):
    cp = cuda_backend.xp
    rng = np.random.default_rng(39)
    # Positive offset with very small fluctuations exercises the reason float64
    # baseline subtraction must remain part of the estimator.
    baseline = rng.uniform(1e9,2e9,(2,*shape))
    pairs = baseline+rng.normal(0,.002,(37,2,*shape))
    pairs[5,0] = np.nan
    pairs[13,1] = np.inf
    if shape:
        baseline[0].flat[0] = np.nan
    cpu,gpu = moments(baseline),moments(baseline,cuda_backend)
    device = cp.asarray(pairs)
    # Interleaved channels plus a reversed sample axis remain strided: no
    # contiguous copy is required by ElementwiseKernel raw-array indexing.
    left,right = device[::-2,0],device[::-2,1]
    host_left,host_right = pairs[::-2,0],pairs[::-2,1]
    gpu.add_many(left,right)
    cpu.add_many(host_left,host_right)
    gpu.add_many(left[:4],right[:4],-1)
    cpu.add_many(host_left[:4],host_right[:4],-1)
    np.testing.assert_array_equal(cp.asnumpy(gpu.sums),cpu.sums)
    actual,count = gpu.result()
    expected,n = cpu.result()
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=2e-15,equal_nan=True)
    np.testing.assert_array_equal(count,n)
    without,nothing = gpu.result(count=False)
    np.testing.assert_array_equal(without,actual)
    assert nothing is None


def test_cuda_minimums_joint_validity_and_nonfinite_results(cuda_backend):
    baseline = np.array([[1.,1.,1.,np.nan,1.,1.,1.],
                         [1.,1.,1.,1.,1.,1.,1.]])
    left = np.array([[2.,0.,-1.,5.,2.,np.nan,1e308],
                     [3.,0.,-1.,6.,np.nan,np.nan,1e308]])
    right = np.array([[4.,2.,2.,5.,4.,3.,1e308],
                      [5.,3.,3.,6.,5.,3.,1e308]])
    cpu,gpu = moments(baseline,min_mean=1.,min_samples=2),moments(baseline,cuda_backend,min_mean=1.,min_samples=2)
    with np.errstate(over="ignore",invalid="ignore"):
        cpu.add_many(left,right)
        gpu.add_many(left,right)
        expected,n = cpu.result()
        actual,count = gpu.result()
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=2e-15,equal_nan=True)
    np.testing.assert_array_equal(count,n)
    assert np.isfinite(actual[0]) and np.isnan(actual[1:]).all()
