"""Direct exports preserve metadata independently of module/source-map helpers."""
from dataclasses import asdict
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eip.domain import SourceSpec
from eip.module_registry import new_node
from eip.readers import open_source
from eip.sequence_processing import process_sequence_job


TOOL = Path(__file__).resolve().parents[1]/"tools"/"direct_cine_exports.py"
_spec = importlib.util.spec_from_file_location("direct_cine_exports", TOOL)
exports = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exports)


def flow(sampling="region"):
    nodes = [new_node(name) for name in exports.ALGORITHMS]
    nodes[2]["parameters"].update(left=dict(x=0,y=0,width=4,height=3),
                                right=dict(x=4,y=0,width=4,height=3), midline=[4,4])
    nodes[3]["parameters"].update(x=1,y=1,width=2,height=2)
    nodes[4]["parameters"].update(mode="mean",output="main",validity="independent")
    nodes[5]["parameters"].update(window=3,map_lag=0,stride=1,analysis="map")
    if sampling == "point":
        point = new_node("point_extract")
        # A nonzero point also checks the existing workbench origin contract:
        # point selection retains the aligned origin rather than cropping it.
        point["parameters"].update(x=1,y=1,output="main")
        nodes = [*nodes[:3],point,nodes[-1]]
    return nodes


@pytest.mark.parametrize("timestamps", [True,False])
@pytest.mark.parametrize("sampling", ["region","point"])
def test_direct_metadata_and_csv_equal_workbench_without_decoding_again(tmp_path,cine_factory,monkeypatch,timestamps,sampling):
    pixels = (np.arange(7*3*8,dtype=np.uint16).reshape(7,3,8)+10)*3
    path,*_ = cine_factory(pixels= pixels,timestamps=timestamps)
    source = SourceSpec(str(path))
    reader = open_source(source)
    payload = dict(id="metadata",display_name="metadata",mode="run",contract_version=3,
                   sources=[asdict(source)], revisions=[reader.meta.revision], steps=flow(sampling),
                   output_dir=str(tmp_path/"workbench"))
    reader.close()
    process_sequence_job(payload,lambda event:None,lambda:False)
    baseline = tmp_path/"workbench"/"run_metadata"/"source_000"
    result = json.loads((baseline.parent/"run.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded",result
    # This is an export-only comparator. The benchmark computes its numerical
    # values independently; this test isolates metadata/CSV generation.
    values = np.load(baseline/"values.npy",allow_pickle=False)
    output = tmp_path/"direct"
    output.mkdir()
    np.save(output/"values.npy", values, allow_pickle=False)
    reader = open_source(source)
    monkeypatch.setattr(reader,"read_frame",lambda index:pytest.fail("exports must not decode pixels"))
    try:
        report = exports.write_exports(output,payload,reader,values,make_plots=False)
    finally:
        reader.close()
    assert (output/"references.jsonl").read_bytes() == (baseline/"references.jsonl").read_bytes()
    assert (output/"measurements.csv").read_bytes() == (baseline/"measurements.csv").read_bytes()
    np.testing.assert_array_equal(np.load(output/"reference_offsets.npy"),np.load(baseline/"reference_offsets.npy"))
    assert report["contract"] == result["results"][0]["contract"]
    assert report["sampling"]["kind"] == sampling
    if sampling == "point":
        assert report["contract"]["meaning"] == "pixel_g2"
        assert report["contract"]["spatial_operation"] is None
        assert report["contract"]["origin"] == [0,0]
    assert report["metadata_pixel_reads"] == 0 and report["reference_records"] == 5


def test_unknown_source_time_is_not_replaced_by_nominal_rate(tmp_path):
    reader = SimpleNamespace(meta=SimpleNamespace(frame_count=5),times=None,ticks=None,
                             source=SimpleNamespace(interval=.01),first_number=-2,trigger_ticks=0)
    payload = dict(id="unknown",steps=flow(),sources=[dict(path="fake.cine",kind="auto",interval=.01)])
    values = np.array([np.nan, 1.2, np.inf],dtype=np.float64)
    np.save(tmp_path/"values.npy",values,allow_pickle=False)
    exports.write_exports(tmp_path,payload,reader,values,make_plots=False)
    rows = [json.loads(line) for line in (tmp_path/"references.jsonl").read_text().splitlines()]
    assert all(row["time"] is None for row in rows)
    assert all(row["correlation"]["known_pairs"] == 0 for row in rows)
    assert all(row["correlation"]["lag_seconds"] is None for row in rows)
    with (tmp_path/"measurements.csv").open(encoding="utf-8-sig",newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows[0]["finite_count"] == "0" and csv_rows[0]["min"] == ""
    assert csv_rows[1]["finite_count"] == "1" and csv_rows[1]["value_1"] == "1.2"
    assert csv_rows[2]["invalid_count"] == "1" and csv_rows[2]["max"] == ""


def test_export_scope_rejects_changed_lag_or_dtype(tmp_path):
    reader = SimpleNamespace(meta=SimpleNamespace(frame_count=5))
    payload = dict(id="scope",steps=flow(),sources=[])
    payload["steps"][-1]["parameters"]["map_lag"] = 1
    with pytest.raises(ValueError,match="zero lag"):
        exports.write_exports(tmp_path,payload,reader,np.ones(3))
    payload["steps"][-1]["parameters"]["map_lag"] = 0
    with pytest.raises(ValueError,match="dtype"):
        exports.write_exports(tmp_path,payload,reader,np.ones(3,dtype=np.float32))


def test_point_and_region_contracts_cannot_be_mislabeled():
    alignment = flow()[2]["parameters"]
    point = dict(kind="point",x=1,y=1)
    contract = exports.output_contract(3,alignment,sampling=point)
    assert contract["meaning"] == "pixel_g2" and contract["origin"] == [0,0]
    wrong = dict(contract,meaning="regional_mean_signal_g2",spatial_operation="mean")
    with pytest.raises(ValueError,match="scalar float64"):
        exports.output_contract(3,alignment,supplied=wrong,sampling=point)
