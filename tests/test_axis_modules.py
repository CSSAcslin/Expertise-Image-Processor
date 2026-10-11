from dataclasses import asdict
import numpy as np
import pytest

from eip.domain import DataSpec, FrameData, FrameRef, SourceSpec
from eip.module_registry import new_node
from eip.module_contracts import compile_flow, source_contract
from eip.sequence_engine import SequenceEngine


class Reader:
    def __init__(self,values,times=None):
        self.values = np.asarray(values)
        self.times_input = times
        self.meta = DataSpec("tiny","sequence",self.values.shape[1:],str(self.values.dtype),"YX",len(values),None,None,"test","gray","tiny")
        self.source = SourceSpec("tiny","sequence")

    def read_frame(self,i):
        return FrameData(self.values[i],FrameRef(i,self.times_input[i] if self.times_input is not None else i*.01))


def node(a,**p):
    n = new_node(a,shape=(2,4),frames=9)
    n["parameters"].update(p)
    return n


def engine(tmp_path,values,nodes,**kw):
    return SequenceEngine(Reader(values),nodes,workdir=tmp_path,**kw)


def test_crop_reduce_and_connect_shapes(tmp_path):
    a = np.arange(3*2*4).reshape(3,2,4)
    nodes = [node("spatial_roi",x=1,width=2,mode="mean"),node("affine",gain=2)]
    e = engine(tmp_path,a,nodes)
    try:
        assert e.output.axes==""
        assert e.output.shape==()
        assert e.get(2,0)==7
        assert e.plan["stages"][0]["output"]["full_axes"]=="T"
    finally:
        e.close()
    with pytest.raises(ValueError,match="Y/X"):
        compile_flow(nodes+[node("gaussian")],source_contract(asdict(Reader(a).meta)))


def test_quantile_invalid_and_colour_axes(tmp_path):
    a = np.array([[[1.,3.],[5.,np.nan]],[[np.nan]*2]*2])
    e = engine(tmp_path,a,[node("spatial_roi",mode="quantile",quantile=.25)])
    try:
        assert e.get(1,0)==2
        assert np.isnan(e.get(1,1))
    finally:
        e.close()
    rgb = np.arange(12).reshape(1,2,2,3)
    e = engine(tmp_path,rgb,[node("spatial_roi",mode="sum")])
    try:
        assert e.output.axes=="C"
        np.testing.assert_array_equal(e.get(1,0),[18,22,26])
    finally:
        e.close()


def test_time_operations_actual_references(tmp_path):
    a = np.arange(10)[:,None,None].astype(float)
    nodes = [node("time_crop",first=2,last=9),node("time_sample",step=2),node("time_smooth",window=3),node("time_bin",size=2)]
    e = engine(tmp_path,a,nodes)
    try:
        assert e.output.length==1
        assert e.get(4,0)==4
        ref = e.ref(4,0)
        assert ref["index"] is None
        assert ref["time"]==pytest.approx(.04)
        assert ref["contribution"]["upstream_stop"]==2
    finally:
        e.close()
    with pytest.raises(ValueError,match="起点"):
        engine(tmp_path,a,[node("affine"),node("time_crop")])


def test_nonmonotonic_real_time_selection_is_exact(tmp_path):
    reader = Reader(np.arange(5.)[:,None,None],times=[0.,2.,.2,3.,.4])
    crop = node("time_crop",unit="seconds",start=.1,end=.5)
    e = SequenceEngine(reader,[crop],workdir=tmp_path)
    try:
        assert e.output.length==2
        assert [e.get(1,j).item() for j in range(2)]==[2.,4.]
        assert [e.ref(1,j)["index"] for j in range(2)]==[2,4]
        assert e.mappings[crop["id"]]["upstream_ranges"]==[[2,3],[4,5]]
    finally:
        e.close()


def test_statistics_uses_edited_input(tmp_path):
    a = np.array([0.,2.,4.,100.])[:,None,None]
    e = engine(tmp_path,a,[node("time_crop",first=2,last=3),node("spatial_roi",mode="mean"),node("statistics",normalization="fixed_rms")])
    try:
        assert e.fit(2)["frames"]==2
        assert e.fit(2)["mean"]==3
        assert e.fit(2)["variance"]==1
        assert e.get(3,1)==4
    finally:
        e.close()


def test_regional_signal_differs_from_mean_pixel_g2(tmp_path):
    a = np.array([[[1.,3.,1.,3.]],[[3.,1.,3.,1.]],[[1.,3.,1.,3.]]])
    align = node("alignment",left=dict(x=0,y=0,width=2,height=1),right=dict(x=2,y=0,width=2,height=1),midline=[2,2])
    corr = node("correlation",window=3,min_samples=2,stride=1,analysis="map",map_lag=0)
    first = engine(tmp_path,a,[align,node("spatial_roi",mode="mean"),corr])
    second = engine(tmp_path,a,[align,corr,node("spatial_roi",mode="mean")])
    try:
        assert first.get(3,0)==1
        expected = ((11/3)/(5/3)**2+(19/3)/(7/3)**2)/2
        assert second.get(3,0)==pytest.approx(expected)
        assert expected>1
    finally:
        first.close()
        second.close()


def test_signed_lags_after_sampling_and_q_is_not_time(tmp_path):
    a = np.zeros((13,1,2))
    a[:,0,0] = np.arange(13)+1
    a[:,0,1] = np.arange(13)**2+1
    align = node("alignment",left=dict(x=0,y=0,width=1,height=1),right=dict(x=1,y=0,width=1,height=1),midline=[1,1])
    nodes = [node("time_sample",step=2),align,node("spatial_roi",mode="mean"),node("correlation",window=3,analysis="curve",lag_list=[-1,0,1])]
    e = engine(tmp_path,a,nodes,center=3)
    try:
        assert e.output.sequence_axis=="Q"
        assert e.output.length==3
        for j,q in enumerate([-1,0,1]):
            left = a[2*np.arange(2,5),0,0]
            right = a[2*(np.arange(2,5)+q),0,1]
            assert e.get(4,j)==pytest.approx(np.mean(left*right)/(left.mean()*right.mean()))
            assert e.ref(4,j)["correlation"]["lag_seconds"]==pytest.approx(q*.02)
    finally:
        e.close()
    with pytest.raises(ValueError,match="横轴是 τ"):
        engine(tmp_path,a,nodes+[node("time_bin")],center=3)


def test_time_edit_after_correlation_trial_keeps_context(tmp_path):
    a = np.zeros((11,1,2))
    a[:,0,0] = np.arange(11)+1
    a[:,0,1] = np.arange(11)**2+1
    align = node("alignment",left=dict(x=0,y=0,width=1,height=1),right=dict(x=1,y=0,width=1,height=1),midline=[1,1])
    nodes = [align,node("correlation",window=3,stride=1),node("time_smooth",window=3)]
    full = engine(tmp_path,a,nodes)
    trial = engine(tmp_path,a,nodes,single=True)
    try:
        assert full.output.length==trial.output.length==7
        np.testing.assert_allclose(full.get(3,0),trial.get(3,0))
        ref = full.ref(3,0)
        assert ref["correlation"]["window"]==3
        assert ref["contribution"]["operation"]=="time_smooth"
    finally:
        full.close()
        trial.close()


def test_joint_roi_support_and_label_semantics(tmp_path):
    a = np.array([[[1.,3.,4.,np.nan]]])
    align = node("alignment",left=dict(x=0,y=0,width=2,height=1),right=dict(x=2,y=0,width=2,height=1),midline=[2,2])
    joint = engine(tmp_path,a,[align,node("spatial_roi",mode="mean")])
    independent = engine(tmp_path,a,[align,node("spatial_roi",mode="mean",validity="independent")])
    try:
        np.testing.assert_array_equal(joint.get(2,0),[1.,4.])
        np.testing.assert_array_equal(independent.get(2,0),[2.,4.])
    finally:
        joint.close()
        independent.close()
    labels = engine(tmp_path,np.arange(12).reshape(3,2,2).astype(np.uint16),[node("threshold",threshold=3),node("time_smooth",window=3),node("spatial_roi",mode="mean")])
    try:
        assert labels.output.semantic=="processed_label"
        assert labels.output.dtype=="float64"
        assert "label" in labels.output.meaning
    finally:
        labels.close()


def test_calibration_declares_integer_and_geometry_dependencies(tmp_path):
    a = np.array([[[10.,10.,20.,20.]]])
    calibrate = node("calibration",black_left=1.,black_right=2.)
    align = node("alignment",left=dict(x=0,y=0,width=2,height=1),right=dict(x=2,y=0,width=2,height=1),midline=[2,2])
    e = engine(tmp_path,a,[calibrate,align])
    try:
        np.testing.assert_array_equal(e.get(2,0),[[[9.,9.]],[[18.,18.]]])
    finally:
        e.close()
    preview = SequenceEngine(Reader(a),[calibrate],workdir=tmp_path,dependencies=[align])
    try:
        np.testing.assert_array_equal(preview.get(1,0),[[9.,9.,18.,18.]])
        assert preview.output.axes=="YX" and len(preview.nodes)==1
        assert preview.plan["preview_dependencies"][0]["id"]==align["id"]
    finally:
        preview.close()
    with pytest.raises(ValueError,match="后续明确配准"):
        compile_flow([calibrate],source_contract(asdict(Reader(a).meta)))
    with pytest.raises(ValueError,match="LUT 需要整数"):
        compile_flow([node("calibration",lut_path="table.npy")],source_contract(asdict(Reader(a).meta)))


def test_plugin_rejects_implicit_dtype_changes(tmp_path):
    import json
    from eip.algorithms import load_manifest
    (tmp_path/"changing.py").write_text("import numpy as np\nclass Changing:\n def prepare(self,parameters,context): pass\n def process(self,pixels,context): return pixels.astype(np.float32 if pixels[0,0]==0 else np.float64)\n def close(self): pass\n",encoding="utf-8")
    path = tmp_path/"changing.json"
    path.write_text(json.dumps(dict(schema_version=1,id="changing",name="Changing",entry_point="changing.py:Changing",parameters=[])),encoding="utf-8")
    plugin = load_manifest(path)
    n = new_node(plugin.id,plugins=[plugin])
    a = np.arange(8,dtype=np.uint16).reshape(2,2,2)
    e = engine(tmp_path,a,[n])
    try:
        assert e.get(1,0).dtype==np.dtype("float32")
        with pytest.raises(ValueError,match="dtype"):
            e.get(1,1)
    finally:
        e.close()
    n["execution"] = dict(precision="float32")
    e = engine(tmp_path,a,[n])
    try:
        assert e.get(1,1).dtype==np.dtype("float32")
    finally:
        e.close()


def test_geometry_shape_compiler_matches_mapping_without_dense_grid():
    from eip.modes.split_correlation.config import GeometryConfig
    from eip.modes.split_correlation.geometry import Geometry,geometry_bounds
    rng = np.random.default_rng(10)
    for mirror_x in (False,True):
        for mirror_y in (False,True):
            for interpolation in ("nearest","bilinear"):
                for _ in range(12):
                    config=GeometryConfig.from_dict(dict(left=dict(x=1,y=0,width=8,height=10),right=dict(x=8,y=1,width=10,height=8),midline=rng.uniform(7,11,2).tolist(),dx=float(rng.uniform(-3,3)),dy=float(rng.uniform(-2,2)),mirror_x=mirror_x,mirror_y=mirror_y,interpolation=interpolation))
                    try:
                        mapping = Geometry(config,(12,18))
                    except ValueError:
                        with pytest.raises(ValueError):
                            geometry_bounds(config,(12,18))
                    else:
                        assert geometry_bounds(config,(12,18))==mapping.crop
