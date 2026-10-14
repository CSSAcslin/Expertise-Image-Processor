"""Frame and paired-frame adapter for the common module registry."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import numpy as np

from eip.algorithms import Pipeline, RunContext
from eip.domain import StepSpec
from eip.module_registry import normalize_nodes


class IdentityPhotometry:
    def apply(self, pixels, side):
        return np.asarray(pixels, dtype=np.float64)


class FrameModules:
    def __init__(self, nodes, shape, context=None, *, input_type="sensor",resources=None):
        from eip.modes.split_correlation.config import GeometryConfig, PhotometryConfig
        from eip.modes.split_correlation.geometry import Geometry
        from eip.modes.split_correlation.photometry import Photometry
        self.context = context or RunContext()
        canonical, self.output_type, self.output_kind = normalize_nodes(nodes, input_type=input_type)
        self.nodes = [n for n in canonical if n["enabled"]]
        if resources:
            from eip.gpu_ops import GPU_MODULES
            from eip.resources import ResourceSettings,MIB
            policy = ResourceSettings.from_dict(resources)
            if int(np.prod(shape))*8*12 > policy.job_mb*MIB:
                raise ValueError("逐帧流程工作数组超过统一任务内存预算，请减少空间范围或提高统一预算")
            for node in self.nodes:
                execution = node.setdefault("execution",{})
                if execution.get("backend","inherit")=="inherit":
                    execution["backend"] = resources["backend"] if node["algorithm"] in GPU_MODULES else "cpu"
                execution["memory_mb"] = min(execution.get("memory_mb") or policy.job_mb,policy.job_mb,int(policy.gpu_mb*.75) if execution["backend"] in {"auto","gpu"} else policy.job_mb)
        self.instances, self.alignments, self.photos = {}, {}, {}
        self.shape, self.output_shape = tuple(shape), tuple(shape)
        self.record = {"modules":deepcopy(self.nodes), "input_shape":list(shape)}
        try:
            for i, node in enumerate(self.nodes):
                algorithm, p = node["algorithm"], node["parameters"]
                if algorithm == "alignment":
                    mapping = Geometry(GeometryConfig.from_dict(p), self.output_shape)
                    self.alignments[i] = mapping
                    self.output_shape = mapping.shape
                elif algorithm == "calibration":
                    self.photos[i] = Photometry(PhotometryConfig.from_dict(dict(p, normalization="none")))
                elif algorithm == "select_channel":
                    pass
                elif algorithm in {"statistics", "correlation"}:
                    raise ValueError("时序模块须使用时序执行器，不能按单帧调用")
                else:
                    self.instances[i] = Pipeline([StepSpec(**node)], self.context)
            self.record.update(output_shape=list(self.output_shape), mappings=[g.record for g in self.alignments.values()], calibrations={self.nodes[i]["id"]:p.record for i,p in self.photos.items()})
            self.record["devices"] = {self.nodes[i]["id"]:dict(actual=getattr(getattr(instance.instances[0],"backend",None),"name","cpu"),reason=getattr(getattr(instance.instances[0],"backend",None),"reason",None)) for i,instance in self.instances.items()}
        except Exception:
            self.close()
            raise

    def execute(self, pixels):
        data, skip = pixels, False
        for i, node in enumerate(self.nodes):
            self.context.check_cancelled()
            if skip:
                skip = False
                continue
            algorithm, p = node["algorithm"], node["parameters"]
            if algorithm == "calibration":
                photo = self.photos[i]
                if i+1 in self.alignments:
                    data = self.alignments[i+1].apply(data, photo)
                    skip = True
                else:
                    mapping = next((g for j,g in self.alignments.items() if j>i), None)
                    if mapping is None:
                        if photo.config.black_left != photo.config.black_right:
                            raise ValueError("单幅校准的左右暗场偏置需相同；不同偏置需有后续分幅模块提供中线")
                        data = photo.apply(data, "left")
                    else:
                        yy = np.arange(data.shape[0])[:,None]
                        line = np.interp(yy, [0,max(1,data.shape[0]-1)], mapping.config.midline)
                        mask = np.arange(data.shape[1])[None,:]<line
                        data = np.where(mask, photo.apply(data,"left"), photo.apply(data,"right"))
            elif algorithm == "alignment":
                data = self.alignments[i].apply(data, IdentityPhotometry())
            elif algorithm == "select_channel":
                data = data[0 if p["channel"]=="left" else 1]
            elif isinstance(data, tuple):
                data = tuple(self.instances[i].execute(a) if node["target"] in {"both",side} else a for a,side in zip(data,("left","right")))
            else:
                scalar = np.ndim(data)==0
                data = self.instances[i].execute(np.asarray(data).reshape(1,1) if scalar else data)
                if scalar:
                    data = data[0,0]
        return data

    def close(self):
        for instance in self.instances.values():
            instance.close()
        self.instances.clear()


class ModulePipeline:
    """Lazy shape adapter; the same frame algorithms serve all workbench modes."""
    def __init__(self, steps, context,resources=None):
        self.resources = resources
        self.nodes, self.output_type, self.output_kind = normalize_nodes([asdict(s) for s in steps])
        if self.output_type in {"pair", "moments"}:
            raise ValueError("流程输出仍为双路数据；请添加「选择幅面」或二阶关联模块")
        self.context, self.engine = context, None

    def execute(self, pixels):
        if self.engine is not None and self.engine.shape!=tuple(pixels.shape):
            self.engine.close()
            self.engine = None
        if self.engine is None:
            self.engine = FrameModules(self.nodes, pixels.shape, self.context,resources=self.resources)
        return self.engine.execute(pixels)

    @property
    def record(self):
        return self.engine.record if self.engine else {"modules":deepcopy(self.nodes)}

    def close(self):
        if self.engine:
            self.engine.close()
