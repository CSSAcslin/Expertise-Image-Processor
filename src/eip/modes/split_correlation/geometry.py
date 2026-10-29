from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from eip.modes.split_correlation.config import GeometryConfig


class RegistrationMethod(Protocol):
    """Replaceable static calibration. Returns right-sampling offsets, not images."""

    def __call__(self, left: np.ndarray, right: np.ndarray) -> dict: ...


def _geometry_extent(config: GeometryConfig, shape):
    """Exact common box and valid count using one-dimensional axis vectors."""
    if len(shape)!=2:
        raise ValueError("分幅关联要求二维灰度矩阵")
    config.left.validate(tuple(shape))
    config.right.validate(tuple(shape))
    if not all(0<x<shape[1] for x in config.midline):
        raise ValueError("中线端点需位于图像内部")
    l,r = config.left,config.right
    x,y = np.arange(l.width,dtype=np.float64),np.arange(l.height,dtype=np.float64)
    xr,yr = x+config.dx,y+config.dy
    if config.mirror_x:
        xr = r.width-1-xr
    if config.mirror_y:
        yr = r.height-1-yr
    if config.interpolation=="nearest":
        xr,yr = np.floor(xr+.5),np.floor(yr+.5)
    vx,vy = (xr>=0)&(xr<=r.width-1),(yr>=0)&(yr<=r.height-1)
    x,xr,y,yr = x[vx],xr[vx],y[vy],yr[vy]
    if not x.size or not y.size:
        raise ValueError("左右幅没有共同有效区域，请检查中线、区域和匹配偏移")
    left_line = np.interp(y+l.y,[0,max(1,shape[0]-1)],config.midline)
    right_line = np.interp(yr+r.y,[0,max(1,shape[0]-1)],config.midline)
    upper = np.searchsorted(x+l.x,left_line,side="left")
    if config.mirror_x:
        lower = np.zeros(y.size,dtype=np.intp)
        upper = np.minimum(upper,np.searchsorted(-(xr+r.x),-right_line,side="right"))
    else:
        lower = np.searchsorted(xr+r.x,right_line,side="left")
    valid = upper>lower
    if not valid.any():
        raise ValueError("左右幅没有共同有效区域，请检查中线、区域和匹配偏移")
    x0,x1 = int(x[lower[valid].min()]),int(x[(upper[valid]-1).max()])+1
    y0,y1 = int(y[valid].min()),int(y[valid].max())+1
    count = int(np.maximum(0,upper-lower).sum(dtype=np.int64))
    return (x0,y0,x1,y1),count


def geometry_bounds(config: GeometryConfig, shape):
    """Exact common bounding box using axis vectors, without a full image grid."""
    return _geometry_extent(config,shape)[0]


@dataclass(frozen=True)
class RegionMapping:
    """Immutable fixed sampler, in absolute sensor coordinates.

    ``bounds`` is relative to the complete aligned output. ``samples`` retains
    the four original bilinear terms and their order; nearest has one term and
    a ``None`` weight. Broadcast axis views keep integer indices compact.
    """

    bounds: tuple[int, int, int, int]
    shape: tuple[int, int]
    mask: np.ndarray
    left_y: np.ndarray
    left_x: np.ndarray
    samples: tuple
    local_samples: tuple
    left_bounds: tuple[int, int, int, int]
    right_bounds: tuple[int, int, int, int]
    left_indices: tuple
    right_indices: tuple
    source_bounds: tuple[int, int, int, int]
    xr: np.ndarray
    yr: np.ndarray


def _readonly(value):
    value.setflags(write=False)
    return value


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
        self.crop, valid_count = _geometry_extent(config,shape)
        x0,y0,x1,y1 = self.crop
        self.shape = (y1-y0,x1-x0)
        self.origin = (config.left.x+x0,config.left.y+y0)
        # A point request need not build any full-size two-dimensional arrays.
        # Retain at most one local mapping and one compatibility/full mapping.
        self._full_mapping = self._local_mapping = None
        self.mapping_builds = 0
        self.record = {"parameters": asdict(config), "reference": "left", "reference_crop": list(self.crop), "origin_in_sensor": list(self.origin), "shape": list(self.shape), "valid_pixels": valid_count, "mapping": "right_local=(left_local+offset), followed by explicit mirrors", "version": "1.0"}

    @property
    def mask(self):
        return self.region_mapping().mask

    @property
    def xr(self):
        return self.region_mapping().xr

    @property
    def yr(self):
        return self.region_mapping().yr

    def region_mapping(self, bounds=None):
        """Compile reusable samples only for an aligned ``(x,y,width,height)``.

        Public shape/origin/record describe the complete logical geometry.
        Mapping arrays are read-only and never reference input frame buffers.
        """
        bounds = (0,0,self.shape[1],self.shape[0]) if bounds is None else tuple(bounds)
        if (len(bounds)!=4 or any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) for v in bounds)
                or min(bounds[0],bounds[1])<0 or min(bounds[2],bounds[3])<1
                or bounds[0]+bounds[2]>self.shape[1] or bounds[1]+bounds[3]>self.shape[0]):
            raise ValueError("配准采样区域超出完整输出范围")
        bounds = tuple(int(v) for v in bounds)
        for cached in (self._full_mapping,self._local_mapping):
            if cached is not None and cached.bounds==bounds:
                return cached
        x,y,w,h = bounds
        l,r = self.config.left,self.config.right
        x0,y0,_,_ = self.crop
        xx = np.arange(x0+x,x0+x+w,dtype=np.float64)[None,:]
        yy = np.arange(y0+y,y0+y+h,dtype=np.float64)[:,None]
        xr,yr = xx+self.config.dx,yy+self.config.dy
        if self.config.mirror_x:
            xr = r.width-1-xr
        if self.config.mirror_y:
            yr = r.height-1-yr
        if self.config.interpolation=="nearest":
            xr,yr = np.floor(xr+.5),np.floor(yr+.5)
        left_line = np.interp(yy+l.y,[0,max(1,self.input_shape[0]-1)],self.config.midline)
        right_line = np.interp(yr+r.y,[0,max(1,self.input_shape[0]-1)],self.config.midline)
        mask = _readonly(((xx+l.x)<left_line)&((xr+r.x)>=right_line)
                         &(xr>=0)&(yr>=0)&(xr<=r.width-1)&(yr<=r.height-1))
        shape = (h,w)
        broadcast = lambda a:_readonly(np.broadcast_to(a,shape))
        left_y,left_x = broadcast(yy.astype(np.intp)+l.y),broadcast(xx.astype(np.intp)+l.x)
        if self.config.interpolation=="nearest":
            samples = ((broadcast(np.clip(yr.astype(np.intp),0,r.height-1)+r.y),
                        broadcast(np.clip(xr.astype(np.intp),0,r.width-1)+r.x),None),)
        else:
            bx,by = np.floor(xr).astype(np.intp),np.floor(yr).astype(np.intp)
            fx,fy = xr-bx,yr-by
            samples = tuple((broadcast(np.clip(by+oy,0,r.height-1)+r.y),
                             broadcast(np.clip(bx+ox,0,r.width-1)+r.x),_readonly(weight))
                            for ox,oy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                                                (0,1,(1-fx)*fy),(1,1,fx*fy)))
        lx0,ly0,lx1,ly1 = int(left_x.min()),int(left_y.min()),int(left_x.max())+1,int(left_y.max())+1
        rx0,ry0 = min(int(sx.min()) for _,sx,_ in samples),min(int(sy.min()) for sy,_,_ in samples)
        rx1,ry1 = max(int(sx.max()) for _,sx,_ in samples)+1,max(int(sy.max()) for sy,_,_ in samples)+1
        min_x,min_y,max_x,max_y = min(lx0,rx0),min(ly0,ry0),max(lx1,rx1),max(ly1,ry1)
        local_samples = tuple((broadcast(sy[:,:1]-r.y),broadcast(sx[:1,:]-r.x),weight)
                              for sy,sx,weight in samples)
        left_indices = (broadcast(left_y[:,:1]-ly0),broadcast(left_x[:1,:]-lx0))
        right_indices = tuple((broadcast(sy[:,:1]-ry0),broadcast(sx[:1,:]-rx0),weight)
                              for sy,sx,weight in samples)
        mapping = RegionMapping(bounds,shape,mask,left_y,left_x,samples,local_samples,
                                (lx0,ly0,lx1-lx0,ly1-ly0),(rx0,ry0,rx1-rx0,ry1-ry0),
                                left_indices,right_indices,
                                (min_x,min_y,max_x-min_x,max_y-min_y),broadcast(xr),broadcast(yr))
        self.mapping_builds += 1
        if bounds==(0,0,self.shape[1],self.shape[0]):
            self._full_mapping = mapping
        else:
            self._local_mapping = mapping
        return mapping

    def apply(self, pixels, photometry):
        return self.apply_region(pixels,photometry)

    def apply_region(self, pixels, photometry, bounds=None):
        """Sample a region while retaining photometry's full-rectangle checks."""
        if tuple(pixels.shape) != self.input_shape:
            raise ValueError("来源帧尺寸发生变化")
        mapping = self.region_mapping(bounds)
        out = np.empty((2,*mapping.shape),dtype=np.float64)
        self.apply_into(pixels,photometry,out,bounds=mapping.bounds)
        return out[0],out[1]

    def apply_into(self, pixels, photometry, out, *, bounds=None):
        """Write one aligned pair into an exclusively owned float64 buffer."""
        if tuple(pixels.shape) != self.input_shape:
            raise ValueError("来源帧尺寸发生变化")
        mapping = self.region_mapping(bounds)
        if tuple(out.shape)!=(2,*mapping.shape) or out.dtype!=np.dtype("float64"):
            raise ValueError("配准输出缓冲区形状或类型不符")
        if not out.flags.writeable or np.may_share_memory(out,pixels):
            raise ValueError("配准输出缓冲区需独占且可写，不能覆盖输入")
        l, r = self.config.left, self.config.right
        left = photometry.apply(pixels[l.y:l.y+l.height, l.x:l.x+l.width], "left")
        right = photometry.apply(pixels[r.y:r.y+r.height, r.x:r.x+r.width], "right")
        x,y,w,h = mapping.bounds
        x0,y0,_,_ = self.crop
        out[0] = left[y0+y:y0+y+h,x0+x:x0+x+w]
        if self.config.interpolation == "nearest":
            sy,sx,_ = mapping.local_samples[0]
            out[1] = right[sy,sx]
        else:
            out[1].fill(0)
            with np.errstate(invalid="ignore"):
                for sy,sx,weight in mapping.local_samples:
                    values = right[sy,sx]
                    out[1] += np.where(weight>0,values*weight,0)
        out[:,~mapping.mask] = np.nan
        return out


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
