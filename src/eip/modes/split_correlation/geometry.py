from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

import numpy as np

from eip.modes.split_correlation.config import GeometryConfig


class RegistrationMethod(Protocol):
    """Replaceable static calibration. Returns right-sampling offsets, not images."""

    def __call__(self, left: np.ndarray, right: np.ndarray) -> dict: ...


class Geometry:
    def __init__(self, config: GeometryConfig, shape):
        if len(shape) != 2:
            raise ValueError("分幅关联目前要求明确的二维灰度强度；请先选择 / 导出灰度通道")
        config.left.validate(tuple(shape))
        config.right.validate(tuple(shape))
        if not all(0 < x < shape[1] for x in config.midline):
            raise ValueError("中线端点需位于图像内部")
        self.config = config
        self.input_shape = tuple(shape)
        l, r = config.left, config.right
        yy, xx = np.indices((l.height, l.width), dtype=np.float64)
        xr, yr = xx + config.dx, yy + config.dy
        if config.mirror_x:
            xr = r.width - 1 - xr
        if config.mirror_y:
            yr = r.height - 1 - yr
        if config.interpolation == "nearest":
            xr, yr = np.floor(xr+.5), np.floor(yr+.5)
        left_line = np.interp(yy + l.y, [0, max(1, shape[0]-1)], config.midline)
        right_line = np.interp(yr + r.y, [0, max(1, shape[0]-1)], config.midline)
        valid = ((xx+l.x) < left_line) & ((xr+r.x) >= right_line) & (xr >= 0) & (yr >= 0) & (xr <= r.width-1) & (yr <= r.height-1)
        vy, vx = np.nonzero(valid)
        if not vx.size:
            raise ValueError("左右幅没有共同有效区域，请检查中线、区域和匹配偏移")
        x0, y0, x1, y1 = int(vx.min()), int(vy.min()), int(vx.max())+1, int(vy.max())+1
        self.crop = (x0, y0, x1, y1)
        self.mask = valid[y0:y1, x0:x1]
        self.xr, self.yr = xr[y0:y1, x0:x1], yr[y0:y1, x0:x1]
        self.shape = self.mask.shape
        self.origin = (l.x+x0, l.y+y0)
        self.record = {"parameters": asdict(config), "reference": "left", "reference_crop": list(self.crop), "origin_in_sensor": list(self.origin), "shape": list(self.shape), "valid_pixels": int(self.mask.sum()), "mapping": "right_local=(left_local+offset), followed by explicit mirrors", "version": "1.0"}

    def apply(self, pixels, photometry):
        if tuple(pixels.shape) != self.input_shape:
            raise ValueError("来源帧尺寸发生变化")
        l, r = self.config.left, self.config.right
        left = photometry.apply(pixels[l.y:l.y+l.height, l.x:l.x+l.width], "left")
        right = photometry.apply(pixels[r.y:r.y+r.height, r.x:r.x+r.width], "right")
        x0, y0, x1, y1 = self.crop
        left = left[y0:y1, x0:x1].copy()
        xr, yr = self.xr, self.yr
        if self.config.interpolation == "nearest":
            right = right[np.clip(yr.astype(int), 0, r.height-1), np.clip(xr.astype(int), 0, r.width-1)]
        else:
            bx, by = np.floor(xr).astype(int), np.floor(yr).astype(int)
            fx, fy = xr-bx, yr-by
            right_out = np.zeros(self.shape, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                for ox, oy, weight in ((0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)), (0,1,(1-fx)*fy), (1,1,fx*fy)):
                    samples = right[np.clip(by+oy,0,r.height-1), np.clip(bx+ox,0,r.width-1)]
                    right_out += np.where(weight > 0, samples*weight, 0)
            right = right_out
        left[~self.mask] = np.nan
        right = np.where(self.mask, right, np.nan)
        return left, right


def phase_registration(left, right):
    """FFT phase translation; fixed calibration, with a quadratic peak refinement."""
    h, w = min(left.shape[0],right.shape[0]), min(left.shape[1],right.shape[1])
    if min(h,w) < 8 or h*w > 4_194_304:
        raise ValueError("自动平移标定需至少 8×8 的区域；大图请缩小有效区")
    def structure(a):
        a = np.asarray(a[:h,:w], dtype=np.float64)
        finite = np.isfinite(a)
        if not finite.any():
            raise ValueError("标定图没有有效像素")
        a = np.where(finite, a-np.mean(a[finite]), 0)
        if np.std(a) < 1e-12:
            raise ValueError("区域缺少结构，不能自动确定匹配；请人工调整")
        return a * np.outer(np.hanning(h),np.hanning(w))
    a, b = structure(left), structure(right)
    product = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    magnitude = np.abs(product)
    normalized = np.divide(product, magnitude, out=np.zeros_like(product), where=magnitude>1e-15)
    correlation = np.fft.ifft2(normalized).real
    py, px = np.unravel_index(np.argmax(correlation),correlation.shape)
    def peak(index, axis):
        n = correlation.shape[axis]
        positions = [(index-1)%n,index,(index+1)%n]
        values = correlation[positions,px] if axis==0 else correlation[py,positions]
        denominator = values[0]-2*values[1]+values[2]
        adjustment = float(np.clip(.5*(values[0]-values[2])/denominator,-.5,.5)) if abs(denominator)>1e-15 else 0.0
        return (index if index <= n//2 else index-n)+adjustment
    dx, dy = -peak(px,1), -peak(py,0)
    confidence = float(correlation[py,px] / max(1e-15,np.sqrt(np.mean(correlation**2))))
    return {"dx": dx, "dy": dy, "peak_to_rms": confidence, "model": "translation", "method": "phase_fft_quadratic", "status": "proposal", "notes": "需用叠加图检查后接受；质量指标不是逐像素误差保证"}


REGISTRATION_METHODS: dict[str, RegistrationMethod] = {"phase": phase_registration}


def propose_rectangles(pixels, midline, threshold=0):
    if pixels.ndim != 2:
        raise ValueError("分幅需二维灰度帧")
    h,w = pixels.shape
    yy,xx = np.indices((h,w))
    line = np.interp(yy,[0,max(1,h-1)],midline)
    regions = []
    for side in (xx < line, xx >= line):
        active = side & np.isfinite(pixels) & (pixels > threshold)
        y,x = np.nonzero(active)
        if not x.size:
            y,x = np.nonzero(side)
        regions.append({"x": int(x.min()),"y": int(y.min()),"width": int(x.max()-x.min()+1),"height": int(y.max()-y.min()+1)})
    return {"left": regions[0], "right": regions[1], "midline": list(midline), "dx": 0, "dy": 0, "mirror_x": False,"mirror_y": False,"interpolation": "nearest"}
