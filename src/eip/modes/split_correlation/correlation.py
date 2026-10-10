from __future__ import annotations

from collections import OrderedDict
from typing import Protocol

import numpy as np


class CorrelationEstimator(Protocol):
    """Extension contract: joint valid count, local means, positive/negative lags."""

    def add(self, left: np.ndarray,right: np.ndarray,sign: int=1): ...
    def result(self): ...


class RollingMoments:
    """Float64 sums of fluctuations, retaining the true local intensity means."""

    def __init__(self, baseline, config, backend):
        self.backend,self.xp = backend,backend.xp
        self.baseline = self.xp.asarray(baseline,dtype=self.xp.float64)
        self.sums = self.xp.zeros((4,*baseline.shape[1:]),dtype=self.xp.float64)
        self.min_mean,self.min_samples = config.min_mean,config.min_samples

    def add(self,left,right,sign=1):
        xp = self.xp
        left,right = xp.asarray(left,dtype=xp.float64),xp.asarray(right,dtype=xp.float64)
        valid = xp.isfinite(left)&xp.isfinite(right)&xp.isfinite(self.baseline[0])&xp.isfinite(self.baseline[1])
        dl = xp.where(valid,left-self.baseline[0],0)
        dr = xp.where(valid,right-self.baseline[1],0)
        self.sums[0] += sign*dl
        self.sums[1] += sign*dr
        self.sums[2] += sign*(dl*dr)
        self.sums[3] += sign*valid

    def result(self):
        xp = self.xp
        n = self.sums[3]
        safe = xp.maximum(n,1)
        ml,mr = self.sums[0]/safe,self.sums[1]/safe
        il,ir = self.baseline[0]+ml,self.baseline[1]+mr
        covariance = self.sums[2]/safe-ml*mr
        valid = (n>=self.min_samples)&(il>self.min_mean)&(ir>self.min_mean)
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            result = xp.where(valid,1+covariance/(il*ir),xp.nan)
        result = xp.where(xp.isfinite(result),result,xp.nan)
        return self.backend.host(result),self.backend.host(n).round().astype(np.uint32)


class PairCache:
    """Bounded transformed-frame cache; its size does not depend on source length."""

    def __init__(self,reader,geometry,photometry,scales,roi,budget,check,*,moment_budget=None):
        self.reader,self.geometry,self.photometry = reader,geometry,photometry
        self.scales,self.roi,self.budget,self.check = scales,roi,budget,check
        self.entries = OrderedDict()
        self.bytes = 0
        self.reads = 0
        self.peak_bytes = 0
        self.moment_budget = moment_budget

    def get(self,index):
        self.check()
        if index in self.entries:
            self.entries.move_to_end(index)
            return self.entries[index]
        frame = self.reader.read_frame(int(index))
        arrays = self.geometry.apply(frame.pixels,self.photometry)
        if self.roi:
            q = self.roi
            arrays = tuple(a[q.y:q.y+q.height,q.x:q.x+q.width].copy() for a in arrays)
        arrays = tuple(a/scale for a,scale in zip(arrays,self.scales))
        size = sum(a.nbytes for a in arrays)
        while self.entries and self.bytes+size>self.budget:
            _,old = self.entries.popitem(last=False)
            self.bytes -= sum(a.nbytes for a in old[0])
        record = (arrays,frame.ref)
        if size <= self.budget:
            self.entries[index] = record
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes,self.bytes)
        self.reads += 1
        return record


def correlate_stream(cache,baseline,config,plan,backend,check,emit,*,inner_progress=None):
    """Lag batches share transformed frames. Rolling windows reread outgoing pairs.

    This bounds RAM even for windows larger than the cache. Non-overlapping windows
    are rebuilt without decoding unused frames. Long updates periodically rebase.
    """
    centers,lags = plan["centers"],plan["lags"]
    pixels = int(np.prod(baseline.shape[1:]))
    moment_budget = cache.moment_budget if cache.moment_budget is not None else int(config.memory_mb*1024**2*.28)
    batch_size = max(1,min(len(lags),moment_budget//max(1,pixels*(48 if backend.name=="gpu" else 32))))
    half = config.window//2
    completed,total = 0,len(centers)*len(lags)
    for batch_start in range(0,len(lags),batch_size):
        batch = list(enumerate(lags[batch_start:batch_start+batch_size],batch_start))
        moments = [RollingMoments(baseline,config,backend) for _ in batch]
        previous = None
        for ordinal,center in enumerate(centers):
            check()
            start,end = center-half,center+half+1
            rebuild = previous is None or center-previous>=config.window or ordinal%256==0
            if rebuild:
                for m in moments:
                    m.sums.fill(0)
                operations = [(range(start,end),1)]
            else:
                operations = [(range(previous-half,start),-1),(range(previous+half+1,end),1)]
            pair_frames = 0
            total_pairs = sum(len(indices) for indices,_ in operations)*len(batch)
            for indices,sign in operations:
                for index in indices:
                    left = cache.get(index)[0][0]
                    for m,(_,lag) in zip(moments,batch):
                        right = cache.get(index+lag)[0][1]
                        m.add(left,right,sign)
                    pair_frames += 1
                    if inner_progress and (pair_frames == 1 or pair_frames%64 == 0):
                        inner_progress(f"窗口 {ordinal+1}/{len(centers)} · 延迟批 {batch_start//batch_size+1}/{(len(lags)+batch_size-1)//batch_size} · 配对累加",pair_frames*len(batch),total_pairs)
            for m,(lag_index,lag) in zip(moments,batch):
                result,count = m.result()
                if hasattr(cache.geometry, "postprocess"):
                    result = cache.geometry.postprocess(result)
                completed += 1
                emit(completed,total)
                yield ordinal,lag_index,center,lag,result,count
            previous = center


def direct_g2(left,right,min_mean=1e-9,min_samples=2):
    """Transparent CPU reference estimator, also useful to external modules."""
    left,right = np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    if left.shape != right.shape or not left.ndim:
        raise ValueError("左右窗口必须具有相同的非空形状")
    valid = np.isfinite(left)&np.isfinite(right)
    n = valid.sum(axis=0)
    safe = np.maximum(n,1)
    l = np.where(valid,left,0)
    r = np.where(valid,right,0)
    ml,mr = l.sum(axis=0)/safe,r.sum(axis=0)/safe
    with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
        g = (l*r).sum(axis=0)/safe/(ml*mr)
    return np.where((n>=min_samples)&(ml>min_mean)&(mr>min_mean)&np.isfinite(g),g,np.nan),n
