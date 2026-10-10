from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from eip.algorithms import sha256_file
from eip.modes.split_correlation.config import PhotometryConfig


class Photometry:
    """Calibrate sensor codes before any spatial interpolation; no implicit clipping."""

    def __init__(self, config: PhotometryConfig):
        self.config = config
        self.lut = None
        self.record = {"parameters": config.__dict__, "lut_sha256": None, "numeric_semantics": config.input_semantics, "version": "1.0"}
        if config.lut_path:
            path = Path(config.lut_path)
            if path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("LUT 文件超过 8 MiB")
            if path.suffix.lower() == ".npy":
                table = np.load(path,allow_pickle=False)
            elif path.suffix.lower() == ".json":
                table = np.asarray(json.loads(path.read_text(encoding="utf-8")))
            else:
                raise ValueError("线性化 LUT 使用一维 .npy 或 JSON 数组")
            if table.ndim != 1 or not 2 <= table.size <= 65536 or table.dtype.kind not in "uif" or not np.isfinite(table).all() or np.any(np.diff(table.astype(np.float64)) < 0):
                raise ValueError("LUT 需为 2–65536 项的有限、单调非减实数数组")
            self.lut = table.astype(np.float64)
            self.record.update(lut_sha256=sha256_file(path),numeric_semantics="user_calibrated_intensity")

    def apply(self, codes, side):
        valid = np.isfinite(codes)
        if self.config.saturation_code is not None:
            valid &= codes < self.config.saturation_code
        if self.lut is not None:
            if codes.dtype.kind not in "uib" or np.any(codes[valid] < 0) or np.any(codes[valid] >= self.lut.size):
                raise ValueError("LUT 仅用于覆盖全部有效整数码值的输入")
            safe = np.where(valid,codes,0).astype(np.int64)
            values = self.lut[safe].copy()
        else:
            values = codes.astype(np.float64)
            if self.config.curve != "identity":
                from .workflow import intensity_curve
                values = intensity_curve(values, self.config.curve, self.config.input_max, self.config.gamma)
        values -= self.config.black_left if side == "left" else self.config.black_right
        values[~valid] = np.nan
        return values


class GlobalStatistics:
    """Per-pixel full-source Welford moments and fixed channel-wide fluctuation RMS."""

    def __init__(self, shape):
        self.count = np.zeros((2,*shape),dtype=np.uint32)
        self.mean = np.zeros((2,*shape),dtype=np.float64)
        self.m2 = np.zeros((2,*shape),dtype=np.float64)

    def add(self, pair):
        for channel, values in enumerate(pair):
            valid = np.isfinite(values)
            old = self.mean[channel]
            self.count[channel] += valid
            delta = np.where(valid,values-old,0)
            update = np.divide(delta,self.count[channel],out=np.zeros_like(delta),where=valid)
            old += update
            self.m2[channel] += np.where(valid,delta*(np.where(valid,values,0)-old),0)

    def finish(self, normalization):
        means = np.where(self.count>0,self.mean,np.nan)
        scales = []
        for channel in range(2):
            denominator = int(self.count[channel].sum(dtype=np.uint64))
            rms = np.sqrt(max(0,float(self.m2[channel].sum())/denominator)) if denominator else 0
            scales.append(float(rms) if normalization == "fixed_rms" and rms > 1e-12 else 1.0)
        return {"mean": means, "count": self.count, "variance": np.divide(self.m2,self.count,out=np.full_like(self.m2,np.nan),where=self.count>0), "scales": np.asarray(scales), "normalization": normalization}
