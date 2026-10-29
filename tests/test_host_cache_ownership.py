"""Cache accounting and ownership for cropped current-work data."""
import gc
import weakref

import numpy as np
import pytest

from eip.sequence_engine import Cache


def assert_accounted(cache):
    assert cache.size == sum(value.nbytes for value in cache.entries.values())
    assert 0 <= cache.size <= cache.budget


def test_replacing_key_releases_old_accounting_and_preserves_other_entries():
    cache = Cache(160)
    cache.put((0,0),np.arange(10,dtype=np.float64))
    cache.put((1,0),np.arange(5,dtype=np.float64))
    replacement = cache.put((0,0),np.arange(6,dtype=np.float64))
    assert cache.get((0,0)) is replacement
    assert cache.get((1,0)) is not None
    assert cache.size == 88
    assert_accounted(cache)


def test_oversized_replacement_removes_stale_key_without_flushing_unrelated_data():
    cache = Cache(80)
    cache.put((0,0),np.ones(4))
    kept = cache.put((1,0),np.ones(4))
    uncached = cache.put((0,0),np.ones(11))
    assert cache.get((0,0)) is None
    assert cache.get((1,0)) is kept
    assert uncached.shape == (11,) and not uncached.flags.writeable
    assert_accounted(cache)


@pytest.mark.parametrize("selection",[lambda a:a[2:5,3:7],lambda a:a[2:3,3:4],lambda a:a[::3,::2].T])
def test_cached_roi_and_point_detach_large_parent(selection):
    parent = np.arange(16*20,dtype=np.float64).reshape(16,20).copy()
    parent_ref = weakref.ref(parent)
    selected = selection(parent)
    expected = selected.copy()
    cache = Cache(expected.nbytes)
    cached = cache.put((3,0),selected)
    assert not np.shares_memory(cached,parent)
    assert cached.flags.owndata and not cached.flags.writeable
    np.testing.assert_array_equal(cached,expected)
    del selected,parent
    gc.collect()
    assert parent_ref() is None
    assert_accounted(cache)


def test_cached_readonly_input_is_never_made_writeable_or_modified():
    parent = np.arange(80,dtype=np.uint16).reshape(8,10).copy()
    parent.setflags(write=False)
    selected = parent[1:4,2:5]
    before = parent.copy()
    cached = Cache(1024).put((2,0),selected)
    assert not cached.flags.writeable
    assert not parent.flags.writeable and not selected.flags.writeable
    with pytest.raises(ValueError):
        cached[0,0] = 999
    np.testing.assert_array_equal(parent,before)


def test_small_external_buffer_selection_does_not_retain_whole_backing_store():
    backing = bytearray(1024*1024)
    value = np.frombuffer(backing,dtype=np.uint8,count=16)
    value[:] = 7
    cached = Cache(16).put((0,0),value)
    assert cached.flags.owndata
    value[:] = 9
    np.testing.assert_array_equal(cached,np.full(16,7,dtype=np.uint8))


def test_cacheable_owned_array_uses_same_allocation():
    owned = np.ones((3,4))
    cache = Cache(owned.nbytes)
    assert cache.put((0,0),owned) is owned
    assert not owned.flags.writeable
    assert_accounted(cache)


def test_eviction_resize_and_discard_stages_obey_byte_budget():
    cache = Cache(96)
    for index in range(3):
        cache.put((index,0),np.full(4,index,dtype=np.float64))
    cache.get((0,0))
    cache.put((3,0),np.ones(4))
    assert cache.get((1,0)) is None
    assert cache.peak == 96
    assert cache.discard_stages({0,3}) == 64
    assert set(cache.entries) == {(2,0)}
    assert_accounted(cache)
    cache.resize(16)
    assert not cache.entries
    assert_accounted(cache)
    cache.resize(64)
    cache.put((4,0),np.ones(4))
    cache.clear()
    assert not cache.entries and cache.size == 0
    assert cache.peak == 96


def test_disabled_cache_and_empty_values_do_not_accumulate_entries():
    cache = Cache(0)
    for index in range(20):
        cache.put((0,index),np.empty(0))
        cache.put((1,index),np.ones(1))
    assert not cache.entries and cache.peak == 0
    assert_accounted(cache)
