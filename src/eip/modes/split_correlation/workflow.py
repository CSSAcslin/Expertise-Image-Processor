"""Sequence adapter for the common, mode-independent module contract."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from eip.algorithms import Pipeline, RunContext
from eip.domain import StepSpec
from eip.module_engine import FrameModules, IdentityPhotometry
from eip.module_ops import intensity_curve
from eip.module_registry import catalog, normalize_nodes
from .geometry import Geometry


MODULES = catalog()


def resolve_workflow(value):
    value["workflow"] = deepcopy(value["workflow"])
    for n in value["workflow"]:
        if n.get("algorithm")=="correlation":
            for key in ("window", "min_mean", "min_samples", "stride", "roi_estimator"):
                if key in value:
                    n.setdefault("parameters",{}).setdefault(key,value[key])
    nodes, state, kind = normalize_nodes(value["workflow"])
    value["workflow"] = nodes
    # Legacy fields are derived compatibility values, never a second parameter owner.
    photo = {"curve":"identity", "black_left":0., "black_right":0., "normalization":"none"}
    for node in nodes:
        if not node["enabled"]:
            continue
        algorithm, p = node["algorithm"], node["parameters"]
        if algorithm == "calibration" and photo.get("instance") is None:
            photo = dict(deepcopy(p), normalization="none", instance=node["id"])
        elif algorithm == "alignment":
            value["geometry"] = deepcopy(p)
        elif algorithm == "statistics":
            photo["normalization"] = p["normalization"]
        elif algorithm == "correlation":
            value.update({key:deepcopy(p[key]) for key in ("window", "min_mean", "min_samples", "stride", "roi_estimator")})
    photo.pop("instance", None)
    value["photometry"] = photo


def has_module(config, algorithm):
    if config.workflow is None:
        return algorithm in {"calibration", "alignment", "statistics", "correlation"}
    return any(n["enabled"] and n["algorithm"] == algorithm for n in config.workflow)


def validate_transform(algorithm, parameters):
    node = {"algorithm":algorithm, "parameters":parameters, "id":"transform"}
    nodes, _, _ = normalize_nodes([node], check_types=False)
    return dict(nodes[0]["parameters"], target=nodes[0]["target"])


def transform_array(values, algorithm, parameters):
    p = deepcopy(parameters)
    p.pop("target", None)
    scalar = np.ndim(values)==0
    with_pipeline = Pipeline([StepSpec(algorithm, p)], RunContext())
    try:
        out = with_pipeline.execute(np.asarray(values).reshape(1,1) if scalar else np.asarray(values))
        return out[0,0] if scalar else out
    finally:
        with_pipeline.close()


class WorkflowGeometry(Geometry):
    def __init__(self, config, shape):
        from .config import GeometryConfig, PhotometryConfig
        from .photometry import Photometry
        self.nodes = [n for n in config.workflow if n["enabled"]]
        stop = next((i for i,n in enumerate(self.nodes) if n["algorithm"] in {"statistics", "correlation"}),len(self.nodes))
        self.upstream = self.nodes[:stop]
        self.downstream = self.nodes[next((i+1 for i,n in enumerate(self.nodes) if n["algorithm"]=="correlation"),len(self.nodes)):]
        first = next((n for n in self.upstream if n["algorithm"]=="alignment"), None)
        if first is None:
            raise ValueError("关联需要对齐双路输入；请添加双幅划分与固定配准模块")
        super().__init__(GeometryConfig.from_dict(first["parameters"]), shape)
        current_shape, mappings = tuple(shape), []
        for n in self.upstream:
            if n["algorithm"]=="alignment":
                g = Geometry(GeometryConfig.from_dict(n["parameters"]), current_shape)
                current_shape = g.shape
                mappings.append(g.record)
        self.shape = current_shape
        self.source_shape = tuple(shape)
        self.record = dict(mappings[-1], workflow=deepcopy(self.upstream), mappings=mappings, version="2.0")
        self.calibrations = {n["id"]:Photometry(PhotometryConfig.from_dict(dict(n["parameters"],normalization="none"))).record for n in self.upstream if n["algorithm"]=="calibration"}
        self.record["calibrations"] = deepcopy(self.calibrations)
        self.engine = self.post_engine = None

    def apply(self, pixels, photometry):
        if self.engine is None:
            self.engine = FrameModules(self.upstream, self.source_shape, getattr(self,"context",None),resources=getattr(self,"resources",None))
            if self.engine.record["calibrations"] != self.calibrations:
                self.engine.close()
                self.engine = None
                raise ValueError("校准文件在运行期间已变化，请重新提交")
        result = self.engine.execute(pixels)
        if not isinstance(result, tuple):
            raise ValueError("关联上游流程必须输出对齐双路")
        return result

    def postprocess(self, values):
        if not self.downstream:
            return values
        if self.post_engine is None:
            self.post_engine = FrameModules(self.downstream, self.shape, getattr(self,"context",None), input_type="map",resources=getattr(self,"resources",None))
        processed = np.asarray(self.post_engine.execute(values), dtype=np.float64)
        return np.where(np.isfinite(values), processed, np.nan)

    def close(self):
        for engine in (self.engine, self.post_engine):
            if engine:
                engine.close()


def build_geometry(config, shape):
    return WorkflowGeometry(config,shape) if config.workflow is not None else Geometry(config.geometry,shape)


def output_semantics(nodes):
    after, post = False, []
    for node in nodes or ():
        if not node.get("enabled", True):
            continue
        if after:
            post.append(node)
        after |= node["algorithm"]=="correlation"
    reference = np.ones((1,1), dtype=float)
    # Arbitrary plugins need not have a meaningful constant-image reference.
    number = None
    if not any(n.get("manifest") for n in post):
        for node in post:
            reference = transform_array(reference,node["algorithm"],node["parameters"])
        number = float(reference[0,0])
    _, _, kind = normalize_nodes(post, input_type="map")
    return {"result_semantics":"processed_g2" if post else "g2", "result_kind":kind, "value_kind":kind, "invalid_policy":"preserve_nan", "coefficient_reference":number if number is not None and math.isfinite(number) else None, "postprocessing":deepcopy(post)}
