"""Changing-lag time summaries reuse actual samples within a hard budget."""
import math

import numpy as np
import pytest

from eip.algorithms import Cancelled
from eip.reference_index import LagTimeIndex, WindowTimeIndex
from test_reference_index import direct, assert_summary, make
from test_axis_modules import node


@pytest.mark.parametrize("budget,mode", [(1024**2, "dense"), (4096, "paged"), (64, "direct")])
def test_signed_irregular_lags_unknown_times_and_random_windows(budget, mode):
    times = [float(i**3*.0001) for i in range(301)]
    times[69] = times[79] = times[101] = None
    metrics = {}
    state = LagTimeIndex(31, len(times), budget, metrics)
    try:
        assert state.mode == mode
        for start, stop, lag in [(70,101,0), (70,101,13), (70,101,-21),
                                 (70,101,13), (132,163,-21), (65,96,1), (70,101,0)]:
            assert_summary(state.summary(start, stop, lag, times.__getitem__, lambda:None),
                           direct(times, start, stop, lag))
        assert metrics["array_bytes"] == state.nbytes <= budget
        assert metrics["peak_array_bytes"] <= budget
        assert metrics["lag_time_peak_reserved_bytes"] <= budget
        assert metrics.get("tree_pair_updates", 0) == 0
    finally:
        state.close()
        state.close()
    assert metrics["array_bytes"] == 0


def test_dense_only_queries_each_required_upstream_time_once():
    length, window = 50654, 10001
    start, stop = 5326, 15327
    lags = [0, 10, 100, 500, 30000, 10, 0]
    reads, metrics = [], {}
    def get_time(index):
        reads.append(index)
        return float(index*.001+(index//13)*.00003)
    state = LagTimeIndex(window, length, 8*1024**2, metrics)
    try:
        for lag in lags:
            result = state.summary(start, stop, lag, get_time, lambda:None)
            assert result[3] == window
            assert result[1] <= result[0]/window <= result[2]
        needed = set(range(start, stop))
        for lag in lags:
            needed.update(range(start+lag, stop+lag))
        assert len(reads) == len(needed)
        assert set(reads) == needed
        assert len(reads) < window*len(lags)
        assert metrics["lag_time_source_queries"] == len(needed)
        assert metrics["pair_time_queries"] == window*len(lags)
        assert state.nbytes <= state.budget
    finally:
        state.close()


def test_paged_cache_does_not_allocate_source_length_and_reuses_pages():
    metrics, reads = {}, []
    def get_time(index):
        reads.append(index)
        return float(index**2*.0001)
    state = LagTimeIndex(31, 100_000_000, 16384, metrics)
    try:
        assert state.mode == "paged"
        first = state.summary(1_000_001,1_000_032,-13,get_time,lambda:None)
        original_reads = len(reads)
        again = state.summary(1_000_001,1_000_032,-13,get_time,lambda:None)
        assert again == first
        assert len(reads) == original_reads
        assert len(state.pages) <= state.page_count <= state._MAX_PAGES
        assert metrics["array_bytes"] <= state.budget
    finally:
        state.close()
    assert not state.pages and metrics["array_bytes"] == 0


def test_tiny_budget_zero_lag_reads_one_side_and_keeps_unknown():
    reads, metrics = [], {}
    def get_time(index):
        reads.append(index)
        return None if index == 6 else index*.01
    state = LagTimeIndex(11, 30, 0, metrics)
    try:
        assert state.summary(3,14,0,get_time,lambda:None) == (0.,0.,0.,10)
        assert reads == list(range(3,14))
        assert state.nbytes == metrics["array_bytes"] == 0
        assert metrics["lag_time_budget_fallback_states"] == 1
        assert metrics["lag_time_direct_windows"] == 1
    finally:
        state.close()


@pytest.mark.parametrize("budget", [1024**2,4096,0])
@pytest.mark.parametrize("bad", [math.nan,math.inf,-math.inf])
def test_present_nonfinite_timestamps_keep_sequential_extrema_order(budget,bad):
    times = [float(i) for i in range(41)]
    times[7],times[11] = bad,None
    metrics = {}
    state = LagTimeIndex(9,len(times),budget,metrics)
    try:
        for start in [3,7,8,6]:
            for lag in [0,1,-2]:
                assert_summary(state.summary(start,start+9,lag,times.__getitem__,lambda:None),
                               direct(times,start,start+9,lag))
        assert metrics["lag_time_direct_windows"] > 0
    finally:
        state.close()


def test_finite_difference_sum_overflow_uses_sequential_fallback():
    # Pair values are finite, but their reduction overflows. Preserve the
    # processor's existing sequential fallback, including known count/extrema.
    times = [0.]*30
    times[10:13] = [1e308]*3
    metrics = {}
    state = LagTimeIndex(3,len(times),1024**2,metrics)
    try:
        assert_summary(state.summary(1,4,9,times.__getitem__,lambda:None), direct(times,1,4,9))
        assert metrics["lag_time_nonfinite_windows"] == 1
    finally:
        state.close()


@pytest.mark.parametrize("budget", [1024**2,4096])
def test_bulk_provider_keeps_none_separate_from_nonfinite_and_avoids_scalar_reads(budget):
    times = [float(i*i*.01) for i in range(201)]
    times[61],times[72],times[83] = None,np.nan,np.inf
    blocks,metrics = [],{}
    def batch(first,stop):
        blocks.append((first,stop))
        return (np.asarray([0. if v is None else v for v in times[first:stop]],np.float64),
                np.asarray([v is not None for v in times[first:stop]],bool))
    def scalar(index):
        raise AssertionError("A valid bulk provider must supply every requested time")
    state = LagTimeIndex(31,len(times),budget,metrics,get_times=batch)
    try:
        for lag in [0,13,-11,17,13]:
            assert_summary(state.summary(60,91,lag,scalar,lambda:None),direct(times,60,91,lag))
        assert blocks
        assert metrics["lag_time_bulk_requests"] == len(blocks)
        assert metrics["lag_time_source_queries"] == sum(b-a for a,b in blocks)
        assert metrics["peak_array_bytes"] <= budget
    finally:
        state.close()


def test_declined_bulk_provider_uses_same_scalar_cache():
    times = np.arange(50,dtype=float)**2*.003
    reads,metrics = [],{}
    def scalar(index):
        reads.append(index)
        return float(times[index])
    state = LagTimeIndex(11,len(times),1024**2,metrics,get_times=lambda a,b:None)
    try:
        result = state.summary(10,21,2,scalar,lambda:None)
        assert_summary(result,direct(times,10,21,2))
        assert len(reads) == 13
        assert state.summary(10,21,2,scalar,lambda:None) == result
        assert len(reads) == 13
    finally:
        state.close()


def test_bad_bulk_dtype_is_error_and_never_publishes_ready_slots():
    state = LagTimeIndex(9,30,1024**2,{},get_times=lambda a,b:(np.ones(b-a,np.float32),np.ones(b-a,bool)))
    try:
        with pytest.raises(ValueError,match="float64"):
            state.summary(2,11,1,lambda i:float(i),lambda:None)
        assert not state.ready.any()
        state.get_times = None
        assert state.summary(2,11,1,lambda i:float(i),lambda:None) == (9.,1.,1.,9)
    finally:
        state.close()


@pytest.mark.parametrize("budget", [1024**2,4096])
def test_cancelled_fill_cannot_reuse_partial_page_or_dense_run(budget):
    state = LagTimeIndex(31,301,budget,{})
    cancelled = False
    def interrupted(index):
        nonlocal cancelled
        if not cancelled and index == 72:
            cancelled = True
            raise Cancelled()
        return float(index*index)
    try:
        with pytest.raises(Cancelled):
            state.summary(60,91,3,interrupted,lambda:None)
        actual = state.summary(60,91,3,interrupted,lambda:None)
        expected = direct([float(i*i) for i in range(301)],60,91,3)
        assert_summary(actual,expected)
        assert state.nbytes <= budget
    finally:
        state.close()


def test_window_larger_than_configured_scratch_remains_bounded():
    times = [i*.001+i*i*.00000003 for i in range(111)]
    state = LagTimeIndex(3,len(times),1024**2,{})
    try:
        assert state.chunk_size == 3
        assert_summary(state.summary(13,91,-3,times.__getitem__,lambda:None),direct(times,13,91,-3))
    finally:
        state.close()


@pytest.mark.parametrize("bounds", [(-1,8,1),(0,0,0),(20,31,0),(0,9,-1),(21,30,1),(0.,9,0)])
def test_invalid_window_is_rejected_before_query(bounds):
    state = LagTimeIndex(9,30,4096,{})
    try:
        with pytest.raises(ValueError):
            state.summary(*bounds,lambda i:pytest.fail("unexpected time query"),lambda:None)
    finally:
        state.close()
    with pytest.raises(RuntimeError):
        state.summary(0,9,0,lambda i:0.,lambda:None)


def test_shared_metric_allocation_released_per_index():
    metrics = {}
    a,b = LagTimeIndex(31,301,1024**2,metrics),LagTimeIndex(17,201,1024**2,metrics)
    assert metrics["array_bytes"] == a.nbytes+b.nbytes
    a.close()
    assert metrics["array_bytes"] == b.nbytes
    b.close()
    assert metrics["array_bytes"] == 0


def test_q_engine_preserves_edited_upstream_times_and_contribution_spans(tmp_path):
    times = [float(i**3*.001) for i in range(64)]
    times[17] = None
    nodes = [node("time_crop",first=2,last=61),node("time_sample",step=2),
             node("time_bin",size=3),node("alignment"),node("time_statistics"),
             node("correlation_curve",window=3,lag_list=[-1,0,2],center=None)]
    reader,engine = make(tmp_path,64,times,nodes)
    groups = [list(range(1+6*j,1+6*j+6,2)) for j in range(10)]
    edited = [float(np.cumsum([times[j] for j in group])[-1]/3) if all(times[j] is not None for j in group) else None for group in groups]
    try:
        for index in [0,1,2,2,0]:
            center,lag,window,start,stop = engine.correlation_position(5,index)
            ref = engine.ref(len(nodes),index)
            summary = direct(edited,start,stop,lag)
            c = ref["correlation"]
            assert ref["sample_index"] == index and ref["time"] == edited[center]
            assert ref["source_span"] == [groups[center][0],groups[center][-1]+1]
            assert ref["contribution"]["operation"] == "time_bin"
            assert c["known_pairs"] == summary[3]
            assert c["lag_seconds"] == (pytest.approx(summary[0]/window) if summary[3] == window else None)
            assert c["lag_seconds_min"] == summary[1] and c["lag_seconds_max"] == summary[2]
        assert isinstance(engine.reference_states[5],LagTimeIndex)
        assert engine.reference_metrics.get("tree_rebuilds",0) == 0
        assert engine.reference_metrics["lag_time_source_queries"] <= len(edited)
        # A sample_ref reader and temporal binning both retain their explicit
        # metadata path rather than bypassing its unknown/contribution rules.
        assert reader.reference_reads > 0
    finally:
        engine.close()
    assert engine.reference_metrics["array_bytes"] == 0
