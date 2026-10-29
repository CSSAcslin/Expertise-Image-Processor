from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from eip.execution_plan import KNOWN_MODULES, capability_for, compile_execution_plan
from eip.module_contracts import Contract
from eip.module_registry import catalog, new_node


def image(axes="YX", shape=(6, 8), sequence_axis="T"):
    return Contract(axes, shape, 12, sequence_axis=sequence_axis)


def node(algorithm, **parameters):
    value = new_node(algorithm)
    value["parameters"].update(parameters)
    return value


def test_every_catalogued_module_has_explicit_dependency_semantics():
    # A newly added module must be reviewed rather than silently inheriting
    # pointwise permissions from its UI category.
    assert set(catalog()) == KNOWN_MODULES
    for algorithm in catalog():
        cap = capability_for(node(algorithm), image())
        assert cap.spatial_dependency != "unknown", algorithm
        assert cap.temporal_dependency != "unknown", algorithm
        json.dumps(cap.record(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        capability_for(node("curve")).domain_pushdown = False


@pytest.mark.parametrize("algorithm,parameters", [
    ("box_blur", {"radius": 2}),
    ("spatial_smooth", {"method": "mean", "radius": 3}),
])
def test_mean_filter_does_not_hide_full_input_validation_or_prefix_rounding(algorithm, parameters):
    cap = capability_for(node(algorithm, **parameters), image())
    assert cap.spatial_dependency == "neighborhood"
    assert cap.spatial_halo == parameters["radius"]
    assert cap.full_input_validation
    assert not cap.domain_pushdown
    assert "integral_order" in cap.barrier_reason


def test_gaussian_has_halo_without_mean_filter_validation():
    cap = capability_for(node("spatial_smooth", method="gaussian", radius=4), image())
    assert cap.domain_pushdown and cap.spatial_halo == 4
    assert not cap.full_input_validation


def test_lut_cannot_skip_code_validation_outside_requested_roi():
    cap = capability_for(node("calibration", lut_path="does-not-need-to-exist.npy"), image())
    assert cap.spatial_dependency == "pointwise"
    assert cap.full_input_validation and not cap.domain_pushdown
    assert "lut" in cap.barrier_reason


def test_split_black_level_requires_sensor_context_before_alignment():
    value = node("calibration", black_left=1., black_right=2.)
    assert not capability_for(value, image()).domain_pushdown
    paired = image("PYX", (2, 6, 8))
    assert capability_for(value, paired).domain_pushdown
    assert capability_for(value, paired.record()).domain_pushdown


def test_scale_broadcast_and_spatially_varying_scale_are_distinguished():
    value = node("fixed_normalize", scale=np.array([1., 2.]).reshape(2, 1, 1))
    paired = image("PYX", (2, 6, 8))
    assert capability_for(value, paired).domain_pushdown
    value["parameters"]["scale"] = np.ones((2, 6, 8))
    assert not capability_for(value, paired).domain_pushdown
    value["parameters"]["scale"] = {"ref": {"node_id": "stats", "port": "value"}}
    assert capability_for(value, paired).barrier_reason == "unresolved_parameter_dependency"


@pytest.mark.parametrize("algorithm", ["time_crop", "time_sample"])
def test_time_selection_keeps_spatial_locality(algorithm):
    cap = capability_for(node(algorithm), image())
    assert cap.temporal_dependency == "selection"
    assert cap.spatial_dependency == "pointwise" and cap.domain_pushdown


def test_axis_slice_uses_actual_sequence_axis_and_line_interpolation_has_halo():
    c = image("PYX", (2, 6, 8), sequence_axis="Q")
    cap = capability_for(node("axis_slice", axis="Q"), c)
    assert cap.temporal_dependency == "selection" and cap.spatial_dependency == "pointwise"
    cap = capability_for(node("axis_slice", axis="Y"), c)
    assert cap.temporal_dependency == "local" and cap.spatial_dependency == "selection"
    cap = capability_for(node("line_extract", mode="segment", interpolation="bilinear"), c)
    assert cap.spatial_halo == 1


def test_full_sequence_statistics_and_automatic_histogram_are_explicit_boundaries():
    for algorithm in ("statistics", "sequence_statistics", "global_statistics"):
        cap = capability_for(node(algorithm), image())
        assert cap.temporal_dependency == "full" and not cap.domain_pushdown
    per_pixel = capability_for(node("global_statistics", scope="samples"), image())
    assert per_pixel.spatial_dependency == "pointwise"
    auto = capability_for(node("distribution", minimum=None), image())
    fixed = capability_for(node("distribution", minimum=0., maximum=100.), image())
    assert auto.temporal_dependency == "full" and not auto.batch
    assert fixed.temporal_dependency == "local" and fixed.batch


def test_plugin_cannot_gain_builtin_permissions_by_reusing_identifier():
    value = dict(algorithm="curve", manifest="missing.json", parameters={})
    cap = capability_for(value, image())
    assert cap.spatial_dependency == cap.temporal_dependency == "unknown"
    assert cap.side_effect and cap.full_input_validation and not cap.domain_pushdown
    assert capability_for(dict(algorithm="future_algorithm")).barrier_reason


def test_plan_preserves_order_devices_precision_disabled_nodes_and_side_contracts():
    first = node("curve")
    disabled = node("calibration")
    disabled["enabled"] = False
    tap = node("data_tap")
    point = node("point_extract", output="side", x=2, y=1)
    point.update(target="left", execution={"backend": "cpu", "precision": "float32"})
    nodes = [first, disabled, tap, point]
    original = deepcopy(nodes)
    contracts = [image()] * 4
    side = dict(node_id=point["id"], stage=2, port="data", contract=Contract("", (), 12))
    devices = {point["id"]: dict(requested="cpu", actual="cpu", reason=None)}
    plan = compile_execution_plan(nodes, contracts, consumers=[side], devices=devices)
    assert nodes == original
    assert [r["node_id"] for r in plan["logical_stages"]] == [n["id"] for n in nodes]
    assert plan["active_module_count"] == 3
    record = plan["logical_stages"][-1]
    assert record["target"] == "left" and record["precision"] == "float32"
    assert record["device"]["actual"] == record["requested_backend"] == "cpu"
    assert plan["logical_stages"][1]["input_stage"] == plan["logical_stages"][1]["output_stage"] == 1
    assert plan["consumers"][0]["contract"]["axes"] == ""
    assert {d["kind"] for d in plan["declared_dependencies"]} == {"side_output", "complete_upstream_delivery"}
    json.dumps(plan, allow_nan=False)
    plan["consumers"][0]["node_id"] = "changed"
    assert side["node_id"] == point["id"]


def test_fingerprint_covers_dependencies_precision_and_device_but_not_display_name():
    source = image()
    value = node("fixed_normalize", scale={"ref": {"node_id": "provider", "port": "value"}})
    base = compile_execution_plan([value], [source, source])
    assert base["declared_dependencies"][0]["node_id"] == "provider"
    value["name"] = "custom display name"
    assert compile_execution_plan([value], [source, source])["fingerprint"] == base["fingerprint"]
    for execution in ({"backend": "gpu"}, {"precision": "float32"}):
        changed = deepcopy(value)
        changed["execution"] = execution
        assert compile_execution_plan([changed], [source, source])["fingerprint"] != base["fingerprint"]
    value["parameters"]["scale"]["ref"]["port"] = "count"
    assert compile_execution_plan([value], [source, source])["fingerprint"] != base["fingerprint"]


def test_contract_count_mismatch_is_rejected_without_resolving_inputs():
    with pytest.raises(ValueError, match="契约数"):
        compile_execution_plan([node("curve")], [image()])
