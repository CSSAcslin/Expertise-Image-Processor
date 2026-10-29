"""Independent timing metadata and bounded, linear moving-window references."""
import math

import numpy as np
import pytest

from eip.algorithms import Cancelled
from eip.reference_index import WindowTimeIndex
from eip.sequence_engine import SequenceEngine
from test_axis_modules import Reader, node


def direct(times, start, stop, lag):
    total, minimum, maximum, known = 0., None, None, 0
    for i in range(start, stop):
        left, right = times[i], times[i+lag]
        if left is not None and right is not None:
            value = right-left
            total += value
            known += 1
            minimum = value if minimum is None else min(minimum, value)
            maximum = value if maximum is None else max(maximum, value)
    return total, minimum, maximum, known


def assert_summary(actual, expected):
    assert actual[3] == expected[3]
    for a, e in zip(actual[:3], expected[:3]):
        if e is None:
            assert a is None
        else:
            assert a == pytest.approx(e, rel=2e-14, abs=2e-14, nan_ok=True)


@pytest.mark.parametrize("lag", [-3, 0, 4])
def test_window_tree_arbitrary_bounds_unknown_times_and_signed_lags(lag):
    times = [float(i**3*.001) for i in range(40)]
    times[8] = times[14] = times[22] = None
    metrics = {}
    state = WindowTimeIndex(17, 1024**2, metrics)
    try:
        for start, stop in [(8,13),(7,15),(10,16),(9,12),(20,26),(19,23),(6,23),(6,23)]:
            assert_summary(state.summary(start,stop,lag,times.__getitem__,lambda:None),
                           direct(times,start,stop,lag))
        # Changing lag invalidates the old timing pairs rather than reusing it.
        assert_summary(state.summary(8,15,-lag,times.__getitem__,lambda:None),
                       direct(times,8,15,-lag))
        assert metrics["array_bytes"] == state.nbytes <= 1024**2
    finally:
        state.close()
    assert metrics["array_bytes"] == 0


def test_sequential_large_windows_only_query_entering_pairs():
    count, window = 6001, 1001
    times = np.arange(count, dtype=float)**2*.00001
    metrics, reads = {}, []
    state = WindowTimeIndex(window, 1024**2, metrics)
    def get_time(i):
        reads.append(i)
        return float(times[i])
    try:
        for start in range(count-window+1):
            result = state.summary(start,start+window,0,get_time,lambda:None)
            assert result == (0.,0.,0.,window)
        assert len(reads) == count
        assert metrics["pair_time_queries"] == count
        assert metrics["tree_rebuilds"] == 1
        assert metrics["tree_pair_updates"] == count + count-window
    finally:
        state.close()


def test_tiny_budget_keeps_direct_reference_without_large_allocation():
    times = [float(i*i) if i != 6 else None for i in range(20)]
    metrics = {}
    state = WindowTimeIndex(11, 64, metrics)
    try:
        assert state.nbytes == 0
        for start in (3,4,2):
            assert_summary(state.summary(start,start+11,1,times.__getitem__,lambda:None),
                           direct(times,start,start+11,1))
        assert metrics["bounded_fallback_windows"] == 3
        assert metrics["array_bytes"] == 0
    finally:
        state.close()


@pytest.mark.parametrize("lag", [0, 1])
def test_nonfinite_time_metadata_retains_direct_order(lag):
    times = [float(i) for i in range(20)]
    times[7] = math.nan
    state = WindowTimeIndex(9,1024**2,{})
    try:
        for start in (3,7,8,6):
            assert_summary(state.summary(start,start+9,lag,times.__getitem__,lambda:None),
                           direct(times,start,start+9,lag))
    finally:
        state.close()


def test_interrupted_update_is_not_reused():
    times = [float(i*i) for i in range(30)]
    state = WindowTimeIndex(9,1024**2,{})
    try:
        state.summary(2,11,1,times.__getitem__,lambda:None)
        def interrupted(i):
            if i == 13:
                raise Cancelled()
            return times[i]
        with pytest.raises(Cancelled):
            state.summary(6,15,1,interrupted,lambda:None)
        assert state.start is None
        assert_summary(state.summary(6,15,1,times.__getitem__,lambda:None),direct(times,6,15,1))
    finally:
        state.close()


class TimedReader(Reader):
    def __init__(self, count, times):
        super().__init__(np.ones((count,2,4)), times)
        self.reference_reads = 0

    def sample_ref(self, index):
        self.reference_reads += 1
        return dict(index=index,time=self.times_input[index],pts=index*7,
                    source_number=index-50,trigger_time=None)


def make(tmp_path, count, times, nodes):
    reader = TimedReader(count,times)
    resources = dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=128,backend="cpu")
    return reader, SequenceEngine(reader,nodes,workdir=tmp_path,resources=resources)


def test_engine_collapses_pass_through_and_linear_zero_lag_references(tmp_path):
    count, window = 3001,1001
    times = [i*.0001 for i in range(count)]
    times[1200] = None
    nodes = [node("time_crop"),node("calibration"),node("alignment"),
             node("roi_crop",width=1,height=1),node("time_statistics"),
             node("correlation_map",window=window,stride=1,map_lag=0)]
    reader, engine = make(tmp_path,count,times,nodes)
    try:
        for index in range(engine.output.length):
            result=engine.ref(len(nodes),index)
            center=index+window//2
            known=window-int(index<=1200<index+window)
            assert result["index"] == center and result["sample_index"] == index
            assert result["time"] == times[center]
            assert result["correlation"] == dict(module_id=nodes[-1]["id"],center_sample=center,
                lag_samples=0,window=window,left_start=index,left_stop=index+window,
                right_start=index,right_stop=index+window,lag_seconds=0. if known==window else None,
                lag_seconds_min=0.,lag_seconds_max=0.,known_pairs=known)
        assert reader.reference_reads == count+engine.output.length
        assert engine.reference_metrics["pair_time_queries"] == count
        assert engine.reference_metrics["source_queries"] == reader.reference_reads
        assert engine.reference_metrics["passthrough_hops_skipped"] > count
        assert engine.reference_metrics["array_bytes"] <= engine.reference_budget
    finally:
        engine.close()


@pytest.mark.parametrize("lag",[-1,0,2])
def test_real_upstream_time_edits_and_source_spans_are_preserved(tmp_path,lag):
    times=[float(i**3*.001) for i in range(64)]
    times[17]=None
    nodes=[node("time_crop",first=2,last=61),node("time_sample",step=2),
           node("time_bin",size=3),node("alignment"),node("time_statistics"),
           node("correlation_map",window=3,stride=1,map_lag=lag)]
    reader,engine=make(tmp_path,64,times,nodes)
    groups=[list(range(1+6*j,1+6*j+6,2)) for j in range(10)]
    # cumsum deliberately retains the legacy sequential time-bin addition,
    # unlike Python 3.12's higher-accuracy built-in sum implementation.
    edited=[float(np.cumsum([times[i] for i in group])[-1]/3) if all(times[i] is not None for i in group) else None
            for group in groups]
    try:
        indices=[*range(engine.output.length),engine.output.length-1,0]
        for index in indices:
            center,q,window,start,stop=engine.correlation_position(5,index)
            result=engine.ref(len(nodes),index)
            expected=direct(edited,start,stop,q)
            correlation=result["correlation"]
            assert result["sample_index"]==index and result["time"]==edited[center]
            assert result["source_span"]==[groups[center][0],groups[center][-1]+1]
            assert result["contribution"]["operation"]=="time_bin"
            assert correlation["lag_seconds"] == (pytest.approx(expected[0]/window) if expected[3]==window else None)
            assert correlation["lag_seconds_min"] == expected[1]
            assert correlation["lag_seconds_max"] == expected[2]
            assert correlation["known_pairs"] == expected[3]
            assert (correlation["left_start"],correlation["right_start"])==(start,start+lag)
    finally:
        engine.close()
