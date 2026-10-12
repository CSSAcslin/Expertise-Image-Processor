from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pytest

from eip.module_registry import catalog, new_node, migrate_workbench_nodes
from eip.module_contracts import compile_flow, source_contract
from eip.sequence_engine import SequenceEngine
from test_axis_modules import Reader


def node(a,**p):
    result=new_node(a,shape=(2,4),frames=9)
    result["parameters"].update(p)
    return result


def make(tmp_path,values,nodes,**kw):
    return SequenceEngine(Reader(values),nodes,workdir=tmp_path,**kw)


def test_catalog_categories_equations_and_canonical_roi():
    specs=catalog()
    assert specs["roi_crop"]["category"]=="intermediate"
    assert {p["key"] for p in specs["roi_crop"]["parameters"]}=={"x","y","width","height"}
    assert specs["correlation_map"]["category"]=="algorithm"
    assert specs["region_reduce"]["category"]=="modifier"
    assert specs["data_tap"]["effect"]=="pass"
    assert specs["affine"]["name"]!=specs["intensity"]["name"]
    assert r"\frac" in specs["correlation_curve"]["formula"]["latex"]
    assert "NaN" in specs["threshold"]["formula"]["validity"]


def test_roi_reduce_migration_preserves_original_and_values(tmp_path):
    old=[node("spatial_roi",x=1,width=2,mode="sum"),node("statistics",normalization="fixed_rms")]
    original=deepcopy(old)
    migrated,report=migrate_workbench_nodes(old)
    assert old==original and len(report)==2
    assert [n["algorithm"] for n in migrated]==["roi_crop","region_reduce","sequence_statistics","fixed_normalize"]
    data=np.arange(24,dtype=np.uint16).reshape(3,2,4)
    first=make(tmp_path,data,old); second=make(tmp_path,data,migrated)
    try:
        for i in range(3):
            np.testing.assert_allclose(first.get(len(first.nodes),i),second.get(len(second.nodes),i))
        assert second.output.dtype=="float64"
    finally:
        first.close();second.close()


def test_sequence_statistics_is_true_passthrough_and_sides_are_single_records(tmp_path):
    data=np.arange(24,dtype=np.uint16).reshape(3,2,4)
    stats=node("sequence_statistics"); engine=make(tmp_path,data,[stats])
    try:
        assert engine.output.dtype=="uint16"
        np.testing.assert_array_equal(engine.get(1,2),data[2])
        sides={s["port"]:s for s in engine.side_outputs()}
        assert set(sides)=={"mean","variance","count","scale"}
        assert sides["mean"]["contract"].sequence_axis=="N"
        assert sides["mean"]["destination"]=="result"
        np.testing.assert_allclose(engine.get_side_output(sides["mean"]["id"],0),data.mean(0))
        np.testing.assert_allclose(engine.get_side_output(sides["variance"]["id"],0),data.var(0))
    finally:
        engine.close()


def test_tap_gets_immediate_upstream_before_downstream_changes(tmp_path):
    data=np.arange(24,dtype=np.uint16).reshape(3,2,4)
    tap=node("data_tap",name="检查裁剪",destination="preview")
    engine=make(tmp_path,data,[node("roi_crop",x=1,width=2),tap,node("affine",gain=10)])
    try:
        side=engine.side_outputs()[0]
        assert side["contract"].shape==(2,2) and side["contract"].dtype=="uint16"
        np.testing.assert_array_equal(engine.get_side_output(side["id"],1),data[1,:,1:3])
        np.testing.assert_allclose(engine.get(3,1),data[1,:,1:3]*10)
        assert engine.side_ref(side["id"],1)["index"]==1
    finally:
        engine.close()


def test_point_and_segment_extraction_side_does_not_replace_main(tmp_path):
    data=np.arange(24,dtype=float).reshape(3,2,4)
    nodes=[node("point_extract",x=2,y=1,output="side"),node("line_extract",mode="segment",x0=0,y0=0,x1=3,y1=1,samples=4,interpolation="bilinear",output="side")]
    engine=make(tmp_path,data,nodes)
    try:
        assert engine.output.axes=="YX"
        point,line=engine.side_outputs()
        assert point["contract"].axes=="" and line["contract"].axes=="R"
        assert engine.get_side_output(point["id"],2)==22
        np.testing.assert_allclose(engine.get_side_output(line["id"],0),np.linspace(0,7,4))
        assert engine.mappings[nodes[1]["id"]]["coordinates"][-1]==pytest.approx(np.sqrt(10))
        np.testing.assert_array_equal(engine.get(2,0),data[0])
    finally:
        engine.close()


def test_axis_line_and_static_slice_use_distinct_R_and_N(tmp_path):
    data=np.arange(24,dtype=float).reshape(3,2,4)
    nodes=[node("line_extract",axis="X",index=1,output="main"),node("axis_slice",axis="T",index=2,output="main")]
    engine=make(tmp_path,data,nodes)
    try:
        assert engine.output.record()["full_axes"]=="NR"
        np.testing.assert_array_equal(engine.get(2,0),data[2,1])
        assert engine.ref(2,0)["time"]==pytest.approx(.02)
    finally:
        engine.close()


def test_distribution_explicit_bins_and_invalid_support(tmp_path):
    data=np.array([[[0,1,2,3],[4,5,np.nan,9]]],dtype=float)
    distribution=node("distribution",bins=2,minimum=0,maximum=4,output="main")
    engine=make(tmp_path,data,[distribution])
    try:
        assert engine.output.axes=="B" and engine.output.semantic=="distribution"
        np.testing.assert_array_equal(engine.get(1,0),[2,3])
        assert engine.mappings[distribution["id"]]["edges"]==[0.,2.,4.]
    finally:
        engine.close()


def test_parameter_binding_resolves_forward_only_and_validates_actual_bounds(tmp_path):
    variable=node("parameter_binding",name="倍率",value=3.)
    transform=node("affine",gain={"ref":{"node_id":variable["id"],"port":"value"}})
    data=np.ones((3,2,4))
    engine=make(tmp_path,data,[variable,transform])
    try:
        np.testing.assert_allclose(engine.get(2,0),3.)
        assert engine.plan["resolved_bindings"][transform["id"]]["gain"]["value"]==3.
    finally:
        engine.close()
    with pytest.raises(ValueError,match="前序"):
        make(tmp_path,data,[transform,variable])
    invalid=node("gaussian",sigma={"ref":{"node_id":variable["id"],"port":"value"}})
    variable["parameters"]["value"]=-1
    with pytest.raises(ValueError,match="范围"):
        make(tmp_path,data,[variable,invalid])


def test_bound_runtime_statistic_is_checked_for_target_bounds(tmp_path):
    stats=node("sequence_statistics")
    invalid=node("curve",gamma={"ref":{"node_id":stats["id"],"port":"mean"}})
    with pytest.raises(ValueError,match="范围"):
        make(tmp_path,-np.ones((3,2,4)),[node("region_reduce"),stats,invalid])


def test_gray_and_bgr_conversion_use_explicit_color_meaning(tmp_path):
    data=np.array([[[[10,20,30],[100,200,250]]]],dtype=np.uint16)
    engine=make(tmp_path,data,[node("color_convert",mode="bgr"),node("grayscale",method="luma")])
    try:
        assert engine.contracts[1].color_order=="BGR" and engine.output.axes=="YX"
        np.testing.assert_allclose(engine.get(2,0),np.sum(data[0]*[.2126,.7152,.0722],axis=-1))
    finally:
        engine.close()


def test_correlation_entries_share_math_and_q_remains_signed(tmp_path):
    data=np.arange(26,dtype=float).reshape(13,1,2)+1
    align=node("alignment",left=dict(x=0,y=0,width=1,height=1),right=dict(x=1,y=0,width=1,height=1),midline=[1,1])
    corr=node("correlation_curve",window=3,lag_list=[-1,0,1])
    engine=make(tmp_path,data,[align,node("region_reduce"),corr],center=6)
    try:
        assert engine.output.sequence_axis=="Q"
        for j,q in enumerate([-1,0,1]):
            left=data[5:8,0,0];right=data[5+q:8+q,0,1]
            assert engine.get(3,j)==pytest.approx(np.mean(left*right)/(left.mean()*right.mean()))
            assert engine.ref(3,j)["correlation"]["lag_samples"]==q
    finally:
        engine.close()
    with pytest.raises(ValueError,match="不是时间"):
        make(tmp_path,data,[align,node("region_reduce"),corr,node("time_sample")])


def test_incompatible_tap_format_and_non_scalar_ref_fail_preflight():
    meta=source_contract(asdict(Reader(np.ones((3,2,4))).meta))
    with pytest.raises(ValueError,match="CSV"):
        compile_flow([node("data_tap",export_format="csv")],meta)
    stats=node("sequence_statistics")
    target=node("affine",gain={"ref":{"node_id":stats["id"],"port":"mean"}})
    with pytest.raises(ValueError,match="标量"):
        compile_flow([stats,target],meta)


def test_custom_numeric_parameter_binding_and_missing_formula_are_honest(tmp_path):
    from eip.algorithms import load_manifest
    (tmp_path/"factor.py").write_text("class Factor:\n def prepare(self,p,c): self.factor=p['factor']\n def process(self,a,c): return a*self.factor\n def close(self): pass\n",encoding="utf-8")
    path=tmp_path/"factor.json"
    path.write_text(json.dumps(dict(schema_version=1,id="custom_factor",name="自定义倍增",entry_point="factor.py:Factor",category="intermediate",parameters=[dict(key="factor",label="倍率",kind="float",default=1.)])),encoding="utf-8")
    spec=load_manifest(str(path)); definition=catalog([spec])[spec.id]
    assert definition["category"]=="intermediate"
    assert definition["formula"]["expression"]=="未提供公式"
    assert definition["bindable_parameters"]==["factor"]
    variable=node("parameter_binding",value=4.)
    custom=new_node(spec.id,plugins=[spec]);custom["parameters"]["factor"]={"ref":{"node_id":variable["id"],"port":"value"}}
    engine=make(tmp_path,np.ones((3,2,4)),[variable,custom])
    try:
        np.testing.assert_array_equal(engine.get(2,0),np.full((2,4),4.))
    finally:
        engine.close()
