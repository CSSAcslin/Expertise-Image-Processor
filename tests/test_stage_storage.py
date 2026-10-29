"""Owned main-path storage: demand filling, strict contracts and cancellation."""
from collections import Counter
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from eip.algorithms import Cancelled
from eip.module_contracts import Contract
from eip.stage_storage import StageStore


def contract(length=9,shape=(2,),dtype="float64"):
    axes = "P" if shape == (2,) else "" if not shape else "YX"
    return Contract(axes,shape,length,dtype=dtype)


def store_for(tmp_path,mode,**kw):
    return StageStore(3,contract(**kw),mode=mode,workdir=tmp_path,block_size=4)


@pytest.mark.parametrize("mode",["ram","disk"])
def test_physical_block_commits_once_without_single_sample_replay(tmp_path,mode):
    values=np.arange(18,dtype=np.float64).reshape(9,2)
    store=store_for(tmp_path,mode)
    calls=[]
    def block(first,stop):
        calls.append((first,stop))
        return values[first:stop]
    def unexpected(index):
        raise AssertionError("A complete physical batch must not be replayed per sample")
    try:
        for index in (1,3,8,4,0,7,6,5,2):
            np.testing.assert_array_equal(store.get(index,unexpected,lambda:None,compute_block=block),values[index])
        assert calls==[(0,4),(8,9),(4,8)]
        assert store.metrics["block_commits"]==3 and store.metrics["built_samples"]==9
        if mode=="disk":
            assert store.metrics["disk_write_bytes"]==values.nbytes
    finally:
        store.close()


@pytest.mark.parametrize("mode",["ram","disk"])
def test_cancelled_physical_commit_does_not_publish_ready_slots(tmp_path,mode):
    values=np.arange(18,dtype=np.float64).reshape(9,2)
    store=store_for(tmp_path,mode)
    checks=0
    def check():
        nonlocal checks
        checks+=1
        if checks==3:
            raise Cancelled("cancel between physical write and readiness")
    try:
        with pytest.raises(Cancelled):
            store.get(0,lambda j:values[j],check,compute_block=lambda a,b:values[a:b])
        assert not store.ready.any() and store.metrics["built_samples"]==0
        np.testing.assert_array_equal(store.get(2,lambda j:values[j],lambda:None,compute_block=lambda a,b:values[a:b]),values[2])
        assert store.ready[:4].all() and store.metrics["built_samples"]==4
    finally:
        store.close()


@pytest.mark.parametrize("mode",["ram","disk"])
def test_demand_blocks_fill_once_and_random_reads_remain_exact(tmp_path,mode):
    values = np.arange(18,dtype=np.float64).reshape(9,2)
    calls,released = [],[]
    store = store_for(tmp_path,mode)

    def compute(index):
        calls.append(index)
        return values[index]

    def release(stop):
        released.append(stop)
        return stop * 10

    try:
        assert store.values is None and store.file is None
        assert not store.ready.any()
        np.testing.assert_array_equal(store.get(6,compute,lambda:None,release),values[6])
        assert calls == [4,5,6,7]
        assert store.ready.tolist() == [False]*4+[True]*4+[False]
        assert released == [8]
        for index in [7,4,6,0,2,8,3,1,5,8]:
            actual = store.get(index,compute,lambda:None,release)
            assert actual.dtype == values.dtype and not actual.flags.writeable
            np.testing.assert_array_equal(actual,values[index])
        assert Counter(calls) == Counter(range(9))
        assert store.ready.all() and store.metrics["state"] == "complete"
        assert released == [8,4,9]
        record = store.record()
        assert record["built_samples"] == 9 and record["hits"] == 8
        assert record["capacity_bytes"] == values.nbytes
        assert record["metadata_bytes"] == 9
        assert record["input_cache_released_bytes"] == 210
        if mode == "disk":
            assert record["disk_write_bytes"] == values.nbytes
            assert record["disk_read_bytes"] == 11 * values[0].nbytes
            assert store.path.stat().st_size == values.nbytes
        else:
            assert store.values.nbytes == values.nbytes
            assert record["disk_write_bytes"] == record["disk_read_bytes"] == 0
    finally:
        store.close()


@pytest.mark.parametrize("mode",["ram","disk"])
@pytest.mark.parametrize("bad",[np.ones(3,dtype=np.float64),np.ones(2,dtype=np.float32)])
def test_wrong_shape_or_dtype_never_commits_a_slot(tmp_path,mode,bad):
    store = store_for(tmp_path,mode)
    try:
        with pytest.raises(ValueError,match="形状或类型"):
            store.get(0,lambda index:bad,lambda:None)
        assert not store.ready.any() and store.metrics["built_samples"] == 0
        values=np.arange(18,dtype=np.float64).reshape(9,2)
        np.testing.assert_array_equal(store.get(0,lambda index:values[index],lambda:None),values[0])
        assert store.ready[:4].all() and store.metrics["built_samples"] == 4
    finally:
        store.close()


@pytest.mark.parametrize("mode",["ram","disk"])
def test_cancelled_fill_keeps_incomplete_slot_unready_and_can_resume(tmp_path,mode):
    values=np.arange(18,dtype=np.float64).reshape(9,2)
    calls,released=[],[]
    interrupted=True
    store=store_for(tmp_path,mode)

    def compute(index):
        calls.append(index)
        if interrupted and index == 1:
            raise Cancelled("test cancellation during computation")
        return values[index]

    try:
        with pytest.raises(Cancelled):
            store.get(0,compute,lambda:None,lambda stop:released.append(stop) or 0)
        assert store.ready.tolist() == [True]+[False]*8
        assert store.metrics["built_samples"] == 1 and released == []
        interrupted=False
        np.testing.assert_array_equal(store.get(1,compute,lambda:None),values[1])
        assert calls == [0,1,1,2,3]
        assert store.ready[:4].all() and store.metrics["built_samples"] == 4
    finally:
        store.close()


@pytest.mark.parametrize("mode",["ram","disk"])
def test_cancellation_before_request_does_not_allocate(tmp_path,mode):
    store=store_for(tmp_path,mode)

    def cancelled():
        raise Cancelled("test cancellation before allocation")

    try:
        with pytest.raises(Cancelled):
            store.get(0,lambda index:pytest.fail("cancelled request computed data"),cancelled)
        assert store.values is None and store.file is None
        assert not store.ready.any() and not store.path.exists()
    finally:
        store.close()


@pytest.mark.parametrize("mode",["ram","disk"])
def test_owned_stage_does_not_retain_large_input_parent(tmp_path,mode):
    parent=np.arange(9*256,dtype=np.float64).reshape(9,256).copy()
    parent_ref=weakref.ref(parent)
    expected=parent[:,7:9].copy()
    store=store_for(tmp_path,mode)
    try:
        for index in range(9):
            actual=store.get(index,lambda j:parent[j,7:9],lambda:None)
            assert not np.shares_memory(actual,parent)
            np.testing.assert_array_equal(actual,expected[index])
        del parent
        gc.collect()
        assert parent_ref() is None
        for index in [8,0,4]:
            np.testing.assert_array_equal(store.get(index,lambda j:pytest.fail("ready slot recomputed"),lambda:None),expected[index])
    finally:
        store.close()


@pytest.mark.parametrize("mode",["ram","disk"])
def test_scalar_contract_and_cleanup(tmp_path,mode):
    store=store_for(tmp_path,mode,length=3,shape=(),dtype="uint64")
    retained=None
    for index in range(3):
        retained=store.get(index,lambda j:np.asarray(j+10,dtype=np.uint64),lambda:None)
        assert retained.shape == () and retained.dtype == np.dtype("uint64")
        assert int(retained) == index+10
    assert store.metrics["capacity_bytes"] == 24
    assert store.metrics["state"] == "complete"
    store.close()
    assert store.values is None and store.ready is None and store.file is None
    assert not store.path.exists()
    assert int(retained) == 12
    store.close()


@pytest.mark.parametrize("mode",["ram","disk"])
@pytest.mark.parametrize("index",[-1,9])
def test_out_of_range_request_never_allocates(tmp_path,mode,index):
    store=store_for(tmp_path,mode)
    try:
        with pytest.raises(IndexError,match="阶段存储样本"):
            store.get(index,lambda j:pytest.fail("invalid request computed data"),lambda:None)
        assert store.values is None and store.file is None and not store.path.exists()
    finally:
        store.close()


def test_disk_allocation_preserves_existing_file(tmp_path):
    path=tmp_path/"stage-3.bin"
    path.write_bytes(b"existing data")
    store=store_for(tmp_path,"disk")
    with pytest.raises(FileExistsError):
        store.get(0,lambda index:np.zeros(2),lambda:None)
    store.close()
    assert path.read_bytes() == b"existing data"


def test_short_disk_write_leaves_slot_unready(tmp_path):
    store=store_for(tmp_path,"disk")
    store._allocate()
    real_file=store.file

    class ShortWriter:
        def seek(self,*args):
            return real_file.seek(*args)

        def write(self,data):
            return real_file.write(data[:-1])

        def close(self):
            return real_file.close()

    store.file=ShortWriter()
    try:
        with pytest.raises(OSError,match="写入不完整"):
            store.get(0,lambda index:np.ones(2),lambda:None)
        assert not store.ready.any() and store.metrics["built_samples"] == 0
        assert store.metrics["disk_write_bytes"] == 0
    finally:
        store.close()
    assert not store.path.exists()


def test_short_disk_read_reports_corruption(tmp_path):
    store=store_for(tmp_path,"disk")
    try:
        store.get(0,lambda index:np.ones(2),lambda:None)
        store.file.truncate(1)
        with pytest.raises(OSError,match="读取不完整"):
            store.get(0,lambda index:pytest.fail("ready slot recomputed"),lambda:None)
    finally:
        store.close()


def range_store(tmp_path, mode, *, length=9, shape=(2,), dtype="float64"):
    # A host device adapter checks the generic owned-view contract without
    # claiming to exercise CUDA. Engine/GPU tests use actual CuPy arrays.
    backend = SimpleNamespace(xp=np) if mode == "device" else None
    return StageStore(3, contract(length, shape, dtype), mode=mode, workdir=tmp_path,
                      backend=backend, block_size=4)


@pytest.mark.parametrize("mode", ["ram", "disk", "device"])
def test_range_fills_each_missing_block_once_and_borrows_owned_storage(tmp_path, mode):
    values = np.arange(18, dtype=np.float64).reshape(9, 2)
    calls, released = [], []
    store = range_store(tmp_path, mode)

    def block(first, stop):
        calls.append((first, stop))
        return values[first:stop]

    def single(index):
        pytest.fail("range blocks must not be replayed per sample")

    def release(stop):
        released.append(stop)
        return stop

    try:
        result = store.get_range(2, 7, single, lambda: None, release,
                                 compute_block=block, max_bytes=values.nbytes)
        np.testing.assert_array_equal(result, values[2:7])
        assert result.dtype == values.dtype and result.flags.c_contiguous
        assert not result.flags.writeable
        assert calls == [(0, 4), (4, 8)] and released == [4, 8]
        assert store.ready.tolist() == [True]*8+[False]
        if mode != "disk":
            assert np.shares_memory(result, store.values)
            assert store.values.flags.writeable
        assert not np.shares_memory(result, values)
        again = store.get_range(3, 5, single, lambda: None, compute_block=block,
                                max_bytes=values.nbytes)
        np.testing.assert_array_equal(again, values[3:5])
        assert calls == [(0, 4), (4, 8)]
        metrics = store.record()
        assert metrics["range_requests"] == 2 and metrics["range_hits"] == 1
        assert metrics["range_samples"] == 7
        assert metrics["hits"] == 0 and metrics["built_samples"] == 8
        assert metrics["input_cache_released_bytes"] == 12
        if mode == "disk":
            assert metrics["disk_write_bytes"] == 8*16
            # One existing get per newly filled physical block, followed by
            # the two contiguous reads. Every byte is included in accounting.
            assert metrics["disk_read_bytes"] == (2+5+2)*16
        final = store.get_range(8, 9, single, lambda: None, compute_block=block)
        np.testing.assert_array_equal(final, values[8:9])
        assert store.metrics["state"] == "complete"
    finally:
        store.close()
    np.testing.assert_array_equal(result, values[2:7])


@pytest.mark.parametrize("mode", ["ram", "disk", "device"])
@pytest.mark.parametrize("index", [0, 9])
def test_empty_range_is_typed_readonly_and_never_allocates_stage(tmp_path, mode, index):
    store = range_store(tmp_path, mode)
    try:
        result = store.get_range(index, index, lambda _:pytest.fail("empty range computed data"),
                                 lambda: None, max_bytes=0)
        assert result.shape == (0, 2) and result.dtype == np.float64
        assert result.nbytes == 0 and not result.flags.writeable
        assert store.values is None and store.file is None and not store.path.exists()
        assert not store.ready.any()
        assert store.metrics["range_requests"] == 1 and store.metrics["range_samples"] == 0
        assert store.metrics["range_hits"] == 0
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["ram", "disk"])
@pytest.mark.parametrize("bounds", [(-1, 2), (1, 10), (4, 3), (1.0, 4), (False, 3)])
def test_invalid_range_does_not_allocate_or_compute(tmp_path, mode, bounds):
    store = range_store(tmp_path, mode)
    try:
        with pytest.raises(IndexError, match="阶段存储范围"):
            store.get_range(*bounds, lambda _:pytest.fail("invalid range computed data"), lambda: None)
        assert store.values is None and store.file is None and not store.path.exists()
        assert store.metrics["range_requests"] == 0
    finally:
        store.close()


def test_disk_range_requires_explicit_budget_beyond_one_physical_block(tmp_path):
    values = np.arange(18, dtype=np.float64).reshape(9, 2)
    store = range_store(tmp_path, "disk")
    calls = []

    def compute(index):
        calls.append(index)
        return values[index]

    try:
        for kw in ({}, dict(max_bytes=79)):
            with pytest.raises(ValueError, match="暂存预算"):
                store.get_range(0, 5, compute, lambda: None, **kw)
        for budget in (-1, 1.0, True):
            with pytest.raises(ValueError, match="非负整数"):
                store.get_range(0, 1, compute, lambda: None, max_bytes=budget)
        assert calls == [] and store.file is None
        assert store.metrics["range_requests"] == 0
        result = store.get_range(0, 5, compute, lambda: None, max_bytes=80)
        np.testing.assert_array_equal(result, values[:5])
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["ram", "device"])
def test_resident_range_does_not_charge_borrowed_view_as_disk_scratch(tmp_path, mode):
    values = np.arange(18, dtype=np.float64).reshape(9, 2)
    store = range_store(tmp_path, mode)
    try:
        result = store.get_range(0, 9, lambda j:values[j], lambda: None, max_bytes=0)
        np.testing.assert_array_equal(result, values)
        assert np.shares_memory(result, store.values)
        assert store.values.flags.writeable
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_range_resumes_partially_ready_block_without_replaying_published_samples(tmp_path, mode):
    values = np.arange(18, dtype=np.float64).reshape(9, 2)
    calls = []
    interrupted = True
    store = range_store(tmp_path, mode)

    def compute(index):
        calls.append(index)
        if interrupted and index == 1:
            raise Cancelled("partly filled physical block")
        return values[index]

    try:
        with pytest.raises(Cancelled):
            store.get(0, compute, lambda: None)
        assert store.ready.tolist() == [True]+[False]*8
        # A fully ready subset must not fill unused neighbours.
        ready = store.get_range(0, 1, compute, lambda: None)
        np.testing.assert_array_equal(ready, values[:1])
        assert calls == [0, 1]
        interrupted = False
        result = store.get_range(0, 3, compute, lambda: None,
                                 compute_block=lambda *_:pytest.fail("cannot replay a partially ready block"))
        np.testing.assert_array_equal(result, values[:3])
        assert calls == [0, 1, 1, 2, 3]
        assert store.ready.tolist() == [True]*4+[False]*5
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_range_cancel_after_physical_write_keeps_only_complete_prior_blocks(tmp_path, mode):
    values = np.arange(18, dtype=np.float64).reshape(9, 2)
    store = range_store(tmp_path, mode)
    calls = []
    last = None
    checks_after_compute = 0
    interrupted = True

    def block(first, stop):
        nonlocal last, checks_after_compute
        last, checks_after_compute = first, 0
        calls.append((first, stop))
        return values[first:stop]

    def check():
        nonlocal checks_after_compute
        if interrupted and last == 4:
            checks_after_compute += 1
            if checks_after_compute == 2:
                raise Cancelled("between physical assignment and readiness")

    try:
        with pytest.raises(Cancelled):
            store.get_range(1, 8, lambda _:pytest.fail("no scalar replay"), check,
                            compute_block=block, max_bytes=values.nbytes)
        assert store.ready.tolist() == [True]*4+[False]*5
        assert store.metrics["built_samples"] == 4
        assert store.metrics["range_samples"] == 0
        interrupted = False
        result = store.get_range(1, 8, lambda _:pytest.fail("no scalar replay"), check,
                                 compute_block=block, max_bytes=values.nbytes)
        np.testing.assert_array_equal(result, values[1:8])
        assert calls == [(0, 4), (4, 8), (4, 8)]
        assert store.metrics["built_samples"] == 8
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_range_cancellation_before_request_does_not_allocate(tmp_path, mode):
    store = range_store(tmp_path, mode)
    try:
        with pytest.raises(Cancelled):
            store.get_range(0, 1, lambda _:pytest.fail("cancelled request computed data"),
                            lambda:(_ for _ in ()).throw(Cancelled("before request")))
        assert store.values is None and store.file is None
        assert store.metrics["range_requests"] == 0
    finally:
        store.close()


def test_range_short_disk_read_does_not_publish_a_successful_range(tmp_path):
    store = range_store(tmp_path, "disk")
    values = np.arange(18, dtype=np.float64).reshape(9, 2)
    try:
        store.get_range(0, 4, lambda j:values[j], lambda: None)
        before = store.metrics["disk_read_bytes"]
        store.file.truncate(1)
        with pytest.raises(OSError, match="范围读取不完整"):
            store.get_range(0, 1, lambda _:pytest.fail("ready data recomputed"), lambda: None)
        assert store.metrics["range_samples"] == 4
        assert store.metrics["disk_read_bytes"] == before
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["ram", "disk", "device"])
def test_range_detaches_input_parent_and_survives_store_close(tmp_path, mode):
    parent = np.arange(9*256, dtype=np.float64).reshape(9, 256).copy()
    parent_ref = weakref.ref(parent)
    expected = parent[:, 7:9].copy()
    store = range_store(tmp_path, mode)
    result = store.get_range(0, 9, lambda j:parent[j, 7:9], lambda: None,
                             compute_block=lambda a,b:parent[a:b, 7:9], max_bytes=expected.nbytes)
    del parent
    gc.collect()
    assert parent_ref() is None
    np.testing.assert_array_equal(result, expected)
    store.close()
    np.testing.assert_array_equal(result, expected)
    assert not result.flags.writeable
    with pytest.raises(RuntimeError, match="已关闭"):
        store.get_range(0, 1, lambda _:pytest.fail("closed store computed data"), lambda: None)


@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_scalar_range_retains_sample_axis_and_exact_dtype(tmp_path, mode):
    store = range_store(tmp_path, mode, length=7, shape=(), dtype="uint64")
    expected = np.arange(10, 17, dtype=np.uint64)
    try:
        actual = store.get_range(np.int64(1), np.int64(6), lambda j:expected[j:j+1].reshape(()),
                                 lambda: None, max_bytes=expected.nbytes)
        np.testing.assert_array_equal(actual, expected[1:6])
        assert actual.shape == (5,) and actual.dtype == np.dtype("uint64")
    finally:
        store.close()
