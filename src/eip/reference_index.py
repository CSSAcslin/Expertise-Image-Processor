"""Bounded window statistics over real, directly-upstream sample times.

T/W queries retain one active signed lag and incrementally update a bounded
window tree. Q scans cache actual upstream times and reduce bounded difference
arrays for each lag. Neither path stores a whole T x Q table.
"""
from __future__ import annotations

import math
from collections import OrderedDict
import numpy as np


class WindowTimeIndex:
    """Float64 sums, exact counts and extrema for an arbitrary moving interval.

The balanced sum can differ from a direct left-to-right sum by rounding. Its
purpose is timing metadata only, not intensity validity or numerical g² state.
Insufficient budget retains the correct direct path without allocating arrays.
"""

    def __init__(self, maximum_window, budget, metrics):
        self.metrics = metrics
        for name in ("pair_time_queries", "tree_rebuilds", "tree_pair_updates",
                     "bounded_fallback_windows", "array_bytes", "peak_array_bytes"):
            metrics.setdefault(name, 0)
        self.start = self.stop = self.lag = None
        size = 1 << (max(1, int(maximum_window))-1).bit_length()
        required = size * 2 * 8 * 4
        # Leave room for the fixed Python state as well as the four arrays.
        self.size = size if required + 1024 <= budget else 0
        self.nbytes = required if self.size else 0
        self.total = self.count = self.minimum = self.maximum = None
        if self.size:
            self.total = np.zeros(size * 2, dtype=np.float64)
            self.count = np.zeros(size * 2, dtype=np.int64)
            self.minimum = np.full(size * 2, np.inf, dtype=np.float64)
            self.maximum = np.full(size * 2, -np.inf, dtype=np.float64)
            metrics["array_bytes"] += self.nbytes
            metrics["peak_array_bytes"] = max(metrics["peak_array_bytes"], metrics["array_bytes"])

    def _pair(self, index, lag, get_time):
        self.metrics["pair_time_queries"] += 1
        left = get_time(index)
        # At zero lag these are the same actual sample; unknown stays unknown.
        right = left if lag == 0 else get_time(index + lag)
        return None if left is None or right is None else right - left

    def _update(self, index, value):
        self.metrics["tree_pair_updates"] += 1
        position = self.size + index % self.size
        self.total[position] = 0. if value is None else value
        self.count[position] = value is not None
        self.minimum[position] = np.inf if value is None else value
        self.maximum[position] = -np.inf if value is None else value
        position //= 2
        while position:
            left, right = position * 2, position * 2 + 1
            self.total[position] = self.total[left] + self.total[right]
            self.count[position] = self.count[left] + self.count[right]
            self.minimum[position] = min(self.minimum[left], self.minimum[right])
            self.maximum[position] = max(self.maximum[left], self.maximum[right])
            position //= 2

    @staticmethod
    def _direct(start, stop, pair, check):
        total, minimum, maximum, known = 0., None, None, 0
        for index in range(start, stop):
            if index % 256 == 0:
                check()
            value = pair(index)
            if value is not None:
                total += value
                known += 1
                minimum = value if minimum is None else min(minimum, value)
                maximum = value if maximum is None else max(maximum, value)
        return total, minimum, maximum, known

    def summary(self, start, stop, lag, get_time, check):
        check()
        if not self.size or stop - start > self.size:
            self.metrics["bounded_fallback_windows"] += 1
            return self._direct(start, stop, lambda j: self._pair(j, lag, get_time), check)
        previous_start, previous_stop = self.start, self.stop
        reuse = (self.lag == lag and previous_start is not None
                 and start < previous_stop and previous_start < stop)
        # A cancelled/failed update must never leave reusable partial state.
        self.start = self.stop = self.lag = None
        if reuse:
            removals = ((previous_start, min(start, previous_stop)),
                        (max(stop, previous_start), previous_stop))
            additions = ((start, min(previous_start, stop)),
                         (max(previous_stop, start), stop))
            for first, last in removals:
                for index in range(first, last):
                    if index % 256 == 0:
                        check()
                    self._update(index, None)
        else:
            self.metrics["tree_rebuilds"] += 1
            self.total.fill(0)
            self.count.fill(0)
            self.minimum.fill(np.inf)
            self.maximum.fill(-np.inf)
            additions = ((start, stop),)
        for first, last in additions:
            for index in range(first, last):
                if index % 256 == 0:
                    check()
                self._update(index, self._pair(index, lag, get_time))
        self.start, self.stop, self.lag = start, stop, lag
        total, known = float(self.total[1]), int(self.count[1])
        if not math.isfinite(total):
            # Preserve legacy order-sensitive nonfinite timestamp handling.
            # Normal real times use the O(1) root query and exact extrema.
            def stored_pair(index):
                position = self.size + index % self.size
                return float(self.total[position]) if self.count[position] else None
            return self._direct(start, stop, stored_pair, check)
        return (total, float(self.minimum[1]) if known else None,
                float(self.maximum[1]) if known else None, known)

    def close(self):
        self.metrics["array_bytes"] -= self.nbytes
        self.nbytes = 0
        self.start = self.stop = self.lag = None
        self.total = self.count = self.minimum = self.maximum = None


class LagTimeIndex:
    """Bounded real-time arrays for a fixed-centre, changing-lag Q scan.

    This cache stores directly-upstream sample times, never differences for an
    entire T x Q grid. Each Q uses bounded NumPy difference/reduction buffers.
    A dense lazy table is used when it fits; otherwise bounded LRU time pages
    reuse samples. Very small budgets keep the correct sequential path.

    ``None`` means unknown; a present NaN/Inf remains present. Unusual
    nonfinite differences or overflowing sums use the legacy sequential order
    for that whole window. These are timing statistics, not intensity sums.
    """

    _FIXED_BYTES = 1024
    _PAGE_OVERHEAD = 256
    _MAX_CHUNK = 16384
    _MAX_PAGES = 64

    def __init__(self, maximum_window, length, budget, metrics, *, get_times=None):
        if (isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer))
                or length < 1 or maximum_window < 1):
            raise ValueError("Lag timing requires positive indexed length and window")
        self.length = int(length)
        self.budget = max(0, int(budget))
        self.metrics, self.get_times = metrics, get_times
        for name in ("array_bytes", "peak_array_bytes", "pair_time_queries",
                     "lag_time_summaries", "lag_time_array_summaries",
                     "lag_time_source_queries", "lag_time_bulk_requests",
                     "lag_time_cache_hits", "lag_time_page_loads",
                     "lag_time_direct_windows", "lag_time_nonfinite_windows",
                     "lag_time_budget_fallback_states", "lag_time_peak_scratch_bytes",
                     "lag_time_peak_reserved_bytes"):
            metrics.setdefault(name, 0)
        self.nbytes, self.chunk_size, self.page_size, self.page_count = 0, 0, 0, 0
        self.mode, self.reason = "direct", "insufficient_time_array_budget"
        self._closed = False
        self.times = self.present = self.ready = None
        self.pages = OrderedDict()
        self._left = self._right = self._lp = self._rp = None
        self._difference = self._known = self._finite = self._missing = None
        available = self.budget-self._FIXED_BYTES
        # Scratch arrays use 29 bytes/sample. Reserve 64 for simultaneously
        # live missing indices, batch-provider results and reduction overhead.
        desired = min(int(maximum_window), self._MAX_CHUNK)
        if available >= self.length*10+desired*64:
            self.mode, self.reason, self.chunk_size = "dense", None, desired
            self.times = np.empty(self.length, np.float64)
            self.present = np.empty(self.length, np.bool_)
            self.ready = np.zeros(self.length, np.bool_)
            self._account(self.times.nbytes+self.present.nbytes+self.ready.nbytes)
        elif available >= 128:
            self.mode, self.reason = "paged", "dense_time_table_exceeds_budget"
            self.chunk_size = min(desired, available//128)
            self.page_size = min(256, self.chunk_size)
            self.page_count = min(self._MAX_PAGES,
                                  (available-self.chunk_size*64)//(self.page_size*9+self._PAGE_OVERHEAD))
        if self.chunk_size:
            size = self.chunk_size
            self._left, self._right = np.empty(size, np.float64), np.empty(size, np.float64)
            self._lp, self._rp = np.empty(size, np.bool_), np.empty(size, np.bool_)
            self._difference = np.empty(size, np.float64)
            self._known, self._finite, self._missing = (np.empty(size, np.bool_) for _ in range(3))
            self._scratch_bytes = sum(a.nbytes for a in (self._left, self._right, self._lp, self._rp,
                                                       self._difference, self._known, self._finite, self._missing))
            self._account(self._scratch_bytes)
            reserved = (self._FIXED_BYTES+size*64+
                        (self.length*10 if self.mode == "dense" else self.page_count*(self.page_size*9+self._PAGE_OVERHEAD)))
            metrics["lag_time_peak_reserved_bytes"] = max(metrics["lag_time_peak_reserved_bytes"], reserved)
            metrics["lag_time_peak_scratch_bytes"] = max(metrics["lag_time_peak_scratch_bytes"], self._scratch_bytes)
        else:
            self._scratch_bytes = 0
            metrics["lag_time_budget_fallback_states"] += 1
        metrics["lag_time_policy"] = {
            "dense": "lazy_dense_upstream_time_arrays",
            "paged": "bounded_upstream_time_pages",
            "direct": "sequential_budget_fallback",
        }[self.mode]

    def _account(self, change):
        self.nbytes += change
        self.metrics["array_bytes"] += change
        self.metrics["peak_array_bytes"] = max(self.metrics["peak_array_bytes"], self.metrics["array_bytes"])

    def _temporary(self, nbytes):
        self.metrics["peak_array_bytes"] = max(self.metrics["peak_array_bytes"], self.metrics["array_bytes"]+nbytes)
        self.metrics["lag_time_peak_scratch_bytes"] = max(self.metrics["lag_time_peak_scratch_bytes"], self._scratch_bytes+nbytes)

    def _fill(self, first, stop, values, present, get_time, check, *, index_bytes=0):
        check()
        if self.get_times is not None:
            self.metrics["lag_time_bulk_requests"] += 1
            batch = self.get_times(first, stop)
            if batch is not None:
                numbers, known = batch
                if (not isinstance(numbers, np.ndarray) or not isinstance(known, np.ndarray)
                        or numbers.shape != (stop-first,) or known.shape != numbers.shape
                        or numbers.dtype != np.dtype("float64") or known.dtype != np.dtype("bool")):
                    raise ValueError("Upstream time block must contain float64 numbers and boolean presence")
                self._temporary(index_bytes+numbers.nbytes+known.nbytes)
                check()
                values[:] = numbers
                present[:] = known
                self.metrics["lag_time_source_queries"] += stop-first
                return
        for index in range(first, stop):
            if (index-first) % 256 == 0:
                check()
            value = get_time(index)
            values[index-first] = 0. if value is None else float(value)
            present[index-first] = value is not None
            self.metrics["lag_time_source_queries"] += 1

    def _ensure_dense(self, first, stop, get_time, check):
        count = stop-first
        if bool(np.all(self.ready[first:stop])):
            self.metrics["lag_time_cache_hits"] += count
            return
        missing_mask = self._missing[:count]
        np.logical_not(self.ready[first:stop], out=missing_mask)
        missing = np.flatnonzero(missing_mask)
        self._temporary(missing.nbytes)
        self.metrics["lag_time_cache_hits"] += count-len(missing)
        if not len(missing):
            return
        # Fill contiguous missing runs. This avoids one source callback per
        # sample for indexed metadata providers without querying cached slots.
        begin = previous = first+int(missing[0])
        for offset in range(1, len(missing)+1):
            current = first+int(missing[offset]) if offset < len(missing) else stop+1
            if current != previous+1:
                end = previous+1
                self._fill(begin, end, self.times[begin:end], self.present[begin:end],
                           get_time, check, index_bytes=missing.nbytes)
                check()
                self.ready[begin:end] = True
                begin = current
            previous = current

    def _page(self, number, get_time, check):
        found = self.pages.get(number)
        if found is not None:
            self.pages.move_to_end(number)
            return found, True
        first = number*self.page_size
        stop = min(self.length, first+self.page_size)
        while self.page_count and len(self.pages) >= self.page_count:
            _, old = self.pages.popitem(last=False)
            self._account(-old[0].nbytes-old[1].nbytes)
            del old
        values, present = np.empty(stop-first, np.float64), np.empty(stop-first, np.bool_)
        self._temporary(values.nbytes+present.nbytes)
        self._fill(first, stop, values, present, get_time, check,
                   index_bytes=values.nbytes+present.nbytes)
        check()
        self.metrics["lag_time_page_loads"] += 1
        if self.page_count:
            self.pages[number] = values, present
            self._account(values.nbytes+present.nbytes)
        return (values, present), False

    def _load(self, first, stop, values, present, get_time, check):
        if self.mode == "dense":
            self._ensure_dense(first, stop, get_time, check)
            values[:] = self.times[first:stop]
            present[:] = self.present[first:stop]
            return
        position = first
        while position < stop:
            check()
            number, offset = divmod(position, self.page_size)
            end = min(stop, (number+1)*self.page_size)
            (page, known), hit = self._page(number, get_time, check)
            amount = end-position
            values[position-first:end-first] = page[offset:offset+amount]
            present[position-first:end-first] = known[offset:offset+amount]
            if hit:
                self.metrics["lag_time_cache_hits"] += amount
            position = end

    def _one(self, index, get_time, check):
        if self.mode == "dense":
            self._ensure_dense(index, index+1, get_time, check)
            return float(self.times[index]) if self.present[index] else None
        if self.mode == "paged":
            number, offset = divmod(index, self.page_size)
            (values, present), hit = self._page(number, get_time, check)
            self.metrics["lag_time_cache_hits"] += int(hit)
            return float(values[offset]) if present[offset] else None
        self.metrics["lag_time_source_queries"] += 1
        return get_time(index)

    def _direct(self, start, stop, lag, get_time, check):
        self.metrics["lag_time_direct_windows"] += 1
        def pair(index):
            left = self._one(index, get_time, check)
            right = left if lag == 0 else self._one(index+lag, get_time, check)
            return None if left is None or right is None else right-left
        return WindowTimeIndex._direct(start, stop, pair, check)

    def summary(self, start, stop, lag, get_time, check):
        if self._closed:
            raise RuntimeError("Lag timing index is closed")
        if (not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                    for v in (start, stop, lag))
                or start < 0 or stop <= start or stop > self.length
                or start+lag < 0 or stop+lag > self.length):
            raise ValueError("Paired time window is outside the upstream sequence")
        start, stop, lag = int(start), int(stop), int(lag)
        check()
        self.metrics["lag_time_summaries"] += 1
        self.metrics["pair_time_queries"] += stop-start
        if not self.chunk_size:
            return self._direct(start, stop, lag, get_time, check)
        self.metrics["lag_time_array_summaries"] += 1
        total, minimum, maximum, known_count = 0., None, None, 0
        for first in range(start, stop, self.chunk_size):
            check()
            end = min(stop, first+self.chunk_size)
            size = end-first
            differences, known, finite = self._difference[:size], self._known[:size], self._finite[:size]
            if self.mode == "dense":
                # The table owns immutable, ready times. These windows need no
                # temporary copies, and filling the right window cannot evict
                # the left as it can with bounded pages.
                self._ensure_dense(first,end,get_time,check)
                if lag:
                    self._ensure_dense(first+lag,end+lag,get_time,check)
                left,right = self.times[first:end],self.times[first+lag:end+lag]
                lp,rp = self.present[first:end],self.present[first+lag:end+lag]
            else:
                left, right = self._left[:size], self._right[:size]
                lp, rp = self._lp[:size], self._rp[:size]
                self._load(first,end,left,lp,get_time,check)
                if lag == 0:
                    right[:] = left
                    rp[:] = lp
                else:
                    self._load(first+lag,end+lag,right,rp,get_time,check)
            np.logical_and(lp, rp, out=known)
            count = int(np.count_nonzero(known))
            with np.errstate(invalid="ignore", over="ignore"):
                np.subtract(right, left, out=differences)
            np.isfinite(differences, out=finite)
            if count != size:
                np.logical_not(known, out=self._missing[:size])
                np.logical_or(finite, self._missing[:size], out=finite)
            if not bool(np.all(finite)):
                self.metrics["lag_time_nonfinite_windows"] += 1
                return self._direct(start, stop, lag, get_time, check)
            if count:
                with np.errstate(invalid="ignore", over="ignore"):
                    partial = float(np.sum(differences,dtype=np.float64) if count == size
                                    else np.sum(differences,where=known,dtype=np.float64))
                total += partial
                if not math.isfinite(total):
                    self.metrics["lag_time_nonfinite_windows"] += 1
                    return self._direct(start, stop, lag, get_time, check)
                low = float(np.min(differences) if count == size else np.min(differences,where=known,initial=np.inf))
                high = float(np.max(differences) if count == size else np.max(differences,where=known,initial=-np.inf))
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
                known_count += count
        return total, minimum, maximum, known_count

    def record(self):
        return dict(policy=self.mode, reason=self.reason, budget_bytes=self.budget,
                    array_bytes=self.nbytes, chunk_samples=self.chunk_size,
                    page_samples=self.page_size, page_capacity=self.page_count)

    def close(self):
        if self._closed:
            return
        self._account(-self.nbytes)
        self.times = self.present = self.ready = None
        self._left = self._right = self._lp = self._rp = None
        self._difference = self._known = self._finite = self._missing = None
        self.pages.clear()
        self.get_times = None
        self._closed = True
