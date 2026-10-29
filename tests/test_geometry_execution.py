"""Fixed samplers must preserve the precompiled geometry's scientific rules."""
from itertools import product

import numpy as np
import pytest

from eip.modes.split_correlation.config import GeometryConfig, PhotometryConfig
from eip.modes.split_correlation.geometry import Geometry
from eip.modes.split_correlation.photometry import Photometry


def config(**changes):
    values = dict(left=dict(x=1,y=1,width=13,height=10),
                  right=dict(x=14,y=0,width=12,height=11),
                  midline=[12.3,15.1],dx=.6,dy=-.7,
                  mirror_x=False,mirror_y=False,interpolation="nearest")
    return GeometryConfig.from_dict(dict(values,**changes))


def dense_reference(config, pixels, photo):
    """Previous dense algorithm, independent of Geometry and its index cache."""
    l,r = config.left,config.right
    yy,xx = np.indices((l.height,l.width),dtype=np.float64)
    xr,yr = xx+config.dx,yy+config.dy
    if config.mirror_x:
        xr = r.width-1-xr
    if config.mirror_y:
        yr = r.height-1-yr
    if config.interpolation=="nearest":
        xr,yr = np.floor(xr+.5),np.floor(yr+.5)
    left_line = np.interp(yy+l.y,[0,max(1,pixels.shape[0]-1)],config.midline)
    right_line = np.interp(yr+r.y,[0,max(1,pixels.shape[0]-1)],config.midline)
    mask = ((xx+l.x)<left_line)&((xr+r.x)>=right_line)&(xr>=0)&(yr>=0)&(xr<=r.width-1)&(yr<=r.height-1)
    vy,vx = np.nonzero(mask)
    x0,y0,x1,y1 = int(vx.min()),int(vy.min()),int(vx.max())+1,int(vy.max())+1
    mask = mask[y0:y1,x0:x1]
    xr,yr = xr[y0:y1,x0:x1],yr[y0:y1,x0:x1]
    left = photo.apply(pixels[l.y:l.y+l.height,l.x:l.x+l.width],"left")[y0:y1,x0:x1].copy()
    right = photo.apply(pixels[r.y:r.y+r.height,r.x:r.x+r.width],"right")
    if config.interpolation=="nearest":
        right = right[np.clip(yr.astype(int),0,r.height-1),np.clip(xr.astype(int),0,r.width-1)]
    else:
        bx,by = np.floor(xr).astype(int),np.floor(yr).astype(int)
        fx,fy = xr-bx,yr-by
        sampled = np.zeros(mask.shape,dtype=np.float64)
        with np.errstate(invalid="ignore"):
            for ox,oy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                                 (0,1,(1-fx)*fy),(1,1,fx*fy)):
                values = right[np.clip(by+oy,0,r.height-1),np.clip(bx+ox,0,r.width-1)]
                sampled += np.where(weight>0,values*weight,0)
        right = sampled
    left[~mask] = np.nan
    right = np.where(mask,right,np.nan)
    return np.stack((left,right)),(x0,y0,x1,y1),int(mask.sum())


@pytest.mark.parametrize("interpolation,mirror_x,mirror_y,dx,dy",list(product(
    ("nearest","bilinear"),(False,True),(False,True),(-1.6,.6),(-.7,1.2))))
def test_compiled_full_and_region_match_dense(interpolation,mirror_x,mirror_y,dx,dy):
    rng = np.random.default_rng(32)
    pixels = rng.uniform(4,800,(12,28))
    pixels[3,2],pixels[7,19],pixels[8,22] = np.nan,np.nan,np.inf
    cfg = config(interpolation=interpolation,mirror_x=mirror_x,mirror_y=mirror_y,dx=dx,dy=dy)
    photo = Photometry(PhotometryConfig(curve="square",input_max=4095,
                                       black_left=.1,black_right=.7,saturation_code=799))
    expected,crop,count = dense_reference(cfg,pixels,photo)
    geometry = Geometry(cfg,pixels.shape)
    assert geometry.crop==crop
    assert geometry.record["valid_pixels"]==count
    assert geometry._full_mapping is None
    h,w = geometry.shape
    bounds = (1,1,max(1,w-2),max(1,h-2))
    x,y,width,height = bounds
    local = geometry.apply_region(pixels,photo,bounds)
    np.testing.assert_array_equal(local,expected[:,y:y+height,x:x+width])
    assert geometry._full_mapping is None
    np.testing.assert_array_equal(geometry.apply(pixels,photo),expected)
    out = np.empty_like(expected)
    assert geometry.apply_into(pixels,photo,out) is out
    np.testing.assert_array_equal(out,expected)
    assert int(geometry.mask.sum())==count


def test_random_sloped_geometry_extent_and_count_match_independent_dense():
    rng = np.random.default_rng(92)
    pixels = rng.uniform(1,200,(12,28))
    photo = Photometry(PhotometryConfig())
    for _ in range(160):
        cfg = config(midline=rng.uniform(6,24,2).tolist(),dx=float(rng.uniform(-6,6)),
                     dy=float(rng.uniform(-5,5)),mirror_x=bool(rng.integers(2)),
                     mirror_y=bool(rng.integers(2)),
                     interpolation="nearest" if rng.integers(2) else "bilinear")
        try:
            expected,crop,count = dense_reference(cfg,pixels,photo)
        except ValueError:
            with pytest.raises(ValueError,match="共同有效"):
                Geometry(cfg,pixels.shape)
        else:
            geometry = Geometry(cfg,pixels.shape)
            assert geometry.crop==crop
            assert geometry.record["valid_pixels"]==count
            np.testing.assert_array_equal(geometry.apply(pixels,photo),expected)


def test_point_mapping_retains_only_requested_grid_and_is_read_only():
    geometry = Geometry(config(),(12,28))
    record = dict(geometry.record)
    mapping = geometry.region_mapping((1,2,1,1))
    assert mapping.shape==(1,1)
    assert geometry.mapping_builds==1
    assert geometry.region_mapping((1,2,1,1)) is mapping
    assert geometry._full_mapping is None
    assert geometry.record==record
    for a in (mapping.mask,mapping.left_y,mapping.left_x,mapping.xr,mapping.yr,
              *(a for sample in mapping.samples for a in sample if a is not None)):
        assert a.shape==(1,1)
        assert not a.flags.writeable
    previous = geometry.region_mapping((0,0,1,1))
    assert geometry._local_mapping is previous
    assert geometry._full_mapping is None


@pytest.mark.parametrize("interpolation",["nearest","bilinear"])
def test_absolute_sensor_sampler_can_read_compact_source(interpolation):
    geometry = Geometry(config(interpolation=interpolation,mirror_x=True,mirror_y=True),(12,28))
    pixels = np.arange(12*28,dtype=np.float64).reshape(12,28)+1
    bounds = (1,1,2,3)
    mapping = geometry.region_mapping(bounds)
    x,y,w,h = mapping.source_bounds
    source = pixels[y:y+h,x:x+w]
    left = source[mapping.left_y-y,mapping.left_x-x].copy()
    right = np.zeros(mapping.shape,dtype=np.float64)
    for sy,sx,weight in mapping.samples:
        values = source[sy-y,sx-x]
        right += values if weight is None else np.where(weight>0,values*weight,0)
    left[~mapping.mask] = right[~mapping.mask] = np.nan
    np.testing.assert_array_equal((left,right),geometry.apply_region(pixels,Photometry(PhotometryConfig()),bounds))
    lx,ly,lw,lh = mapping.left_bounds
    rx,ry,rw,rh = mapping.right_bounds
    left_source = pixels[ly:ly+lh,lx:lx+lw]
    right_source = pixels[ry:ry+rh,rx:rx+rw]
    left = left_source[mapping.left_indices].copy()
    right = np.zeros(mapping.shape,dtype=np.float64)
    for sy,sx,weight in mapping.right_indices:
        values = right_source[sy,sx]
        right += values if weight is None else np.where(weight>0,values*weight,0)
    left[~mapping.mask] = right[~mapping.mask] = np.nan
    np.testing.assert_array_equal((left,right),geometry.apply_region(pixels,Photometry(PhotometryConfig()),bounds))


@pytest.mark.parametrize("interpolation",["nearest","bilinear"])
def test_repeated_frames_do_not_rebuild_or_clip_indices(monkeypatch,interpolation):
    geometry = Geometry(config(interpolation=interpolation),(12,28))
    geometry.region_mapping()
    pixels = np.arange(12*28,dtype=np.float64).reshape(12,28)
    photo = Photometry(PhotometryConfig())
    expected = geometry.apply(pixels,photo)
    def no_clip(*args,**kwargs):
        raise AssertionError("fixed indices must have been clipped during compilation")
    monkeypatch.setattr(np,"clip",no_clip)
    for _ in range(3):
        np.testing.assert_array_equal(geometry.apply(pixels,photo),expected)
    assert geometry.mapping_builds==1


def test_zero_bilinear_weights_do_not_propagate_neighbor_nan():
    geometry = Geometry(config(dx=0.,dy=0.,interpolation="bilinear",midline=[14,14]),(12,28))
    pixels = np.ones((12,28),dtype=np.float64)
    pixels[1,15] = np.nan
    _,right = geometry.apply_region(pixels,Photometry(PhotometryConfig()),(0,0,1,1))
    assert right[0,0]==1


def test_region_keeps_full_photometry_validation(tmp_path):
    path = tmp_path/"lut.npy"
    np.save(path,np.arange(16,dtype=np.float64)**2)
    geometry = Geometry(config(dx=0,dy=0),(12,28))
    pixels = np.ones((12,28),dtype=np.uint16)
    # Outside requested output, but inside an original photometry rectangle.
    pixels[9,10] = 19
    with pytest.raises(ValueError,match="LUT"):
        geometry.apply_region(pixels,Photometry(PhotometryConfig(lut_path=str(path))),(0,0,1,1))


def test_apply_into_preserves_input_and_rejects_alias_and_bad_shape():
    geometry = Geometry(config(),(12,28))
    pixels = np.arange(12*28,dtype=np.float64).reshape(12,28)
    before = pixels.copy()
    photo = Photometry(PhotometryConfig())
    out = np.empty((2,*geometry.shape),dtype=np.float64)
    geometry.apply_into(pixels,photo,out)
    np.testing.assert_array_equal(pixels,before)
    with pytest.raises(ValueError,match="尺寸"):
        geometry.apply(pixels[:-1],photo)
    with pytest.raises(ValueError,match="类型"):
        geometry.apply_into(pixels,photo,np.empty(out.shape,dtype=np.float32))
    with pytest.raises(ValueError,match="独占"):
        geometry.apply_into(pixels,photo,pixels[:2,:1].reshape(2,1,1),bounds=(0,0,1,1))
    for bounds in ((-1,0,1,1),(0,0,0,1),(0,0,100,1),(0.,0,1,1),(True,0,1,1)):
        with pytest.raises(ValueError,match="区域"):
            geometry.region_mapping(bounds)
