"""Describe module dependencies without reading pixels or selecting devices.

Capabilities describe legal execution transformations, not a second algorithm
configuration. In particular ``batch`` means independent samples may be grouped
by a suitable adapter; it does not permit passing a batch axis to an existing
single-sample function. Numerical kernels, validation and precision boundaries
remain the responsibility of the ordinary module implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from collections.abc import Mapping


PLAN_SCHEMA_VERSION = 1
CAPABILITY_VERSION = 1


def dense_signal_schedule(length, window, lag, centers):
    """Do not eagerly materialize a mostly unused signal for sparse windows."""
    if not length or not isinstance(centers,range) or not centers or centers.step > window:
        return False
    first = centers.start-(window-1)//2
    stop = centers[-1]-(window-1)//2+window
    overlap = max(0,min(stop,stop+lag)-max(first,first+lag))
    covered = 2*(stop-first)-overlap
    return covered*4 >= int(length)*3


@dataclass(frozen=True)
class ExecutionCapability:
    spatial_dependency: str
    temporal_dependency: str = "local"
    spatial_halo: int = 0
    full_input_validation: bool = False
    side_effect: bool = False
    batch: bool = False
    domain_pushdown: bool = False
    read_only_alias: bool = False
    exclusive_write: bool = False
    barrier_reason: str | None = None
    notes: tuple[str, ...] = ()

    def record(self):
        value = asdict(self)
        value["notes"] = list(self.notes)
        return value


POINTWISE = frozenset({"intensity", "affine", "gain_offset", "curve", "threshold",
                       "fixed_normalize", "grayscale", "color_convert"})
SELECTIONS = frozenset({"roi_crop", "spatial_roi", "select_channel", "point_extract",
                        "line_extract", "axis_slice", "paired_trace"})
TEMPORAL_SELECTIONS = frozenset({"time_crop", "time_sample"})
TEMPORAL_WINDOWS = frozenset({"time_smooth", "time_bin"})
CORRELATIONS = frozenset({"correlation", "correlation_map", "correlation_curve", "correlation_window"})
KNOWN_MODULES = (POINTWISE | SELECTIONS | TEMPORAL_SELECTIONS | TEMPORAL_WINDOWS |
                 CORRELATIONS | frozenset({"calibration", "alignment", "statistics",
                 "sequence_statistics", "time_statistics", "region_reduce", "global_statistics", "distribution",
                 "parameter_binding", "data_tap", "spatial_smooth", "box_blur", "gaussian"}))


def _has_reference(value):
    if isinstance(value, Mapping):
        return set(value) == {"ref"} or any(_has_reference(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_reference(v) for v in value)
    return False


def _contract_field(contract, name, default=None):
    return contract.get(name, default) if isinstance(contract, Mapping) else getattr(contract, name, default)


def _spatial_scale(parameters, contract):
    """True when a resolved scale varies across a spatial input dimension."""
    scale = parameters.get("scale", 1.)
    shape = getattr(scale, "shape", None)
    if shape is None:
        # Public scale parameters are scalar; resolved references may be arrays.
        return isinstance(scale, (tuple, list))
    axes = _contract_field(contract, "axes", "")
    if not shape:
        return False
    if not axes or len(shape) > len(axes):
        return True
    padded = (1,) * (len(axes) - len(shape)) + tuple(shape)
    return any(padded[k] != 1 for k, axis in enumerate(axes) if axis in "YX")


def capability_for(node, input_contract=None):
    """Return conservative, parameter-aware dependencies for one module.

    Inputs may be a normal module node or the same node with resolved numeric
    parameters. Unknown/plugin implementations never inherit built-in powers
    from an algorithm identifier. No manifest, LUT or input file is opened.
    """
    a, p = node.get("algorithm", ""), node.get("parameters", {})
    if node.get("manifest") or a not in KNOWN_MODULES:
        return ExecutionCapability("unknown", "unknown", full_input_validation=True,
                                   side_effect=True, barrier_reason="undeclared_plugin_or_module")
    local = ExecutionCapability("pointwise", batch=True, domain_pushdown=True)
    if a in POINTWISE:
        cap = replace(local, exclusive_write=a in {"gain_offset", "affine", "fixed_normalize"},
                      notes=("preserve_instance_precision_and_target",))
        if a == "fixed_normalize" and _spatial_scale(p, input_contract):
            cap = replace(cap, domain_pushdown=False,
                          barrier_reason="spatial_scale_requires_matching_domain",
                          notes=cap.notes + ("validate_entire_scale_before_slicing",))
    elif a == "calibration":
        cap = replace(local, notes=("calibrate_each_pixel_before_interpolation_or_reduction",))
        if p.get("lut_path"):
            cap = replace(cap, full_input_validation=True, domain_pushdown=False,
                          barrier_reason="lut_requires_full_input_code_validation")
        elif p.get("black_left", 0.) != p.get("black_right", 0.) and not _contract_field(input_contract, "paired", False):
            # Contract.paired is a property; dict contracts expose their axes.
            if not _contract_field(input_contract, "axes", "").startswith("P"):
                cap = replace(cap, domain_pushdown=False,
                              barrier_reason="split_black_level_requires_sensor_midline")
    elif a == "alignment":
        cap = ExecutionCapability("selection", batch=True, domain_pushdown=True,
                                  spatial_halo=1 if p.get("interpolation") == "bilinear" else 0,
                                  notes=("fixed_sampling_map_and_original_invalid_mask",
                                         "calibration_precedes_interpolation"))
    elif a in {"box_blur", "gaussian", "spatial_smooth"}:
        mean = a == "box_blur" or a == "spatial_smooth" and p.get("method", "mean") == "mean"
        cap = ExecutionCapability("neighborhood", spatial_halo=int(p.get("radius", 1)),
                                  full_input_validation=mean, batch=True,
                                  domain_pushdown=not mean,
                                  barrier_reason="mean_filter_full_finite_check_and_integral_order" if mean else None,
                                  notes=("preserve_original_boundary_and_validity_weights",))
    elif a in SELECTIONS:
        temporal = "local"
        spatial = "selection"
        alias = a in {"select_channel", "point_extract"}
        halo = 1 if a == "line_extract" and p.get("mode") == "segment" and p.get("interpolation") == "bilinear" else 0
        if a == "axis_slice":
            sequence_axis = _contract_field(input_contract, "sequence_axis", None)
            if p.get("axis") == sequence_axis or sequence_axis is None and p.get("axis") in {"T", "Q", "W", "N"}:
                spatial, temporal = "pointwise", "selection"
        cap = ExecutionCapability(spatial, temporal, spatial_halo=halo, batch=True,
                                  domain_pushdown=True, read_only_alias=alias,
                                  notes=("selection_is_relative_to_direct_input",))
    elif a in TEMPORAL_SELECTIONS:
        cap = replace(local, temporal_dependency="selection", read_only_alias=True,
                      notes=("retain_actual_sample_time_and_source_mapping",))
    elif a in TEMPORAL_WINDOWS:
        cap = replace(local, temporal_dependency="window", batch=False,
                      notes=("retain_window_state_and_upstream_sample_units",))
    elif a in {"time_statistics", "region_reduce"}:
        cap = ExecutionCapability("full", batch=True,
                                  barrier_reason="spatial_reduction_consumes_complete_direct_input",
                                  notes=("preserve_finite_reduction_order_and_joint_or_independent_rule",))
    elif a == "global_statistics":
        cap = ExecutionCapability("pointwise" if p.get("scope") == "samples" else "full", "full",
                                  barrier_reason="global_statistic_consumes_complete_declared_scope",
                                  notes=("static_N_output_is_not_time",))
    elif a in {"statistics", "sequence_statistics"}:
        cap = ExecutionCapability("full", "full", barrier_reason="full_sequence_statistic_provider",
                                  notes=("fixed_rms_is_not_spatial_standard_deviation",))
    elif a == "distribution":
        auto = any(p.get(key) is None for key in ("minimum", "maximum"))
        cap = ExecutionCapability("full", "full" if auto else "local", batch=not auto,
                                  barrier_reason="histogram_edges_require_full_sequence" if auto else "histogram_consumes_complete_direct_input")
    elif a in CORRELATIONS:
        curve = a == "correlation_curve" or a == "correlation" and p.get("analysis", "map") == "curve"
        cap = ExecutionCapability("full" if curve else "pointwise", "window",
                                  barrier_reason="correlation_requires_axis_specific_state",
                                  notes=("float64_common_valid_pairs_signed_lag",
                                         "W_expansion_and_Q_scan_do_not_use_sliding_T_state"))
    elif a == "data_tap":
        cap = ExecutionCapability("pointwise", side_effect=True, read_only_alias=True,
                                  barrier_reason="data_tap_keeps_complete_upstream_delivery")
    else:  # parameter_binding publishes a value while its main stream is identity.
        cap = replace(local, read_only_alias=True, batch=False,
                      notes=("keep_parameter_provider_and_all_referenced_ports",))
    if _has_reference(p):
        cap = replace(cap, domain_pushdown=False,
                      barrier_reason="unresolved_parameter_dependency",
                      notes=cap.notes + ("resolve_and_validate_binding_before_domain_planning",))
    return cap


def _record(value):
    if hasattr(value, "record"):
        return _record(value.record())
    if isinstance(value, Mapping):
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _references(value):
    if isinstance(value, Mapping):
        if set(value) == {"ref"} and isinstance(value["ref"], Mapping):
            yield dict(value["ref"])
        else:
            for item in value.values():
                yield from _references(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _references(item)


def compile_execution_plan(nodes, contracts, *, consumers=(), devices=None):
    """Build an immutable-by-copy description of a validated logical flow.

    ``contracts`` contains the source followed by each enabled module output.
    Disabled instances remain visible. ``consumers`` accepts actual side-output
    records from the engine; these are preserved, including their contracts.
    Numerical execution and resource admission are deliberately not performed.
    """
    nodes = list(nodes)
    contracts = list(contracts)
    count = sum(bool(node.get("enabled", True)) for node in nodes)
    if len(contracts) != count + 1:
        raise ValueError("执行计划契约数需等于启用模块数加来源")
    records, dependencies, barriers = [], [], []
    stage = 0
    for node in nodes:
        enabled = bool(node.get("enabled", True))
        before = contracts[stage]
        cap = capability_for(node, before)
        before_stage = stage
        if enabled:
            stage += 1
        device = (devices or {}).get(node.get("id"), {})
        record = dict(node_id=node.get("id"), algorithm=node.get("algorithm"),
                      enabled=enabled, input_stage=before_stage,
                      output_stage=stage, input=_record(before), output=_record(contracts[stage]),
                      capability=cap.record(), requested_backend=node.get("execution", {}).get("backend", "inherit"),
                      device=_record(device), precision=node.get("execution", {}).get("precision", "native"),
                      target=node.get("target", "both"))
        records.append(record)
        if not enabled:
            continue
        if cap.barrier_reason:
            barriers.append(dict(node_id=node.get("id"), reason=cap.barrier_reason))
        for reference in _references(node.get("parameters", {})):
            dependencies.append(dict(consumer_node_id=node.get("id"), kind="parameter_reference", **reference))
        if node.get("algorithm") == "data_tap":
            dependencies.append(dict(consumer_node_id=node.get("id"), kind="complete_upstream_delivery", stage=before_stage))
        elif node.get("parameters", {}).get("output") == "side":
            dependencies.append(dict(consumer_node_id=node.get("id"), kind="side_output", stage=before_stage))
        elif node.get("algorithm") in {"sequence_statistics", "statistics"}:
            dependencies.append(dict(consumer_node_id=node.get("id"), kind="full_sequence_statistic", stage=before_stage))
    payload = dict(nodes=[{key: _record(value) for key, value in node.items() if key != "name"} for node in nodes],
                   contracts=[_record(contract) for contract in contracts],
                   consumers=_record(list(consumers)), devices=_record(devices or {}),
                   capability_version=CAPABILITY_VERSION)
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return dict(schema_version=PLAN_SCHEMA_VERSION, capability_version=CAPABILITY_VERSION,
                logical_module_count=len(nodes), active_module_count=count,
                logical_stages=records, consumers=payload["consumers"],
                declared_dependencies=dependencies, barriers=barriers,
                fingerprint=fingerprint)
