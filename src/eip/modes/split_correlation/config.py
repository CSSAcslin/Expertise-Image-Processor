from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from eip.domain import ROI


def integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name}必须是{'不小于 ' + str(minimum) + ' 的' if minimum is not None else ''}整数")
    return value


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}必须是有限数值")
    return float(value)


@dataclass(frozen=True)
class GeometryConfig:
    left: ROI
    right: ROI
    midline: tuple[float, float]
    dx: float = 0.0
    dy: float = 0.0
    mirror_x: bool = False
    mirror_y: bool = False
    interpolation: str = "nearest"

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["left"] = ROI(**value["left"])
        value["right"] = ROI(**value["right"])
        line = value["midline"]
        if not isinstance(line, (list, tuple)) or len(line) != 2:
            raise ValueError("中线需指定顶部和底部的 X 坐标")
        value["midline"] = tuple(finite(x, "中线坐标") for x in line)
        result = cls(**value)
        finite(result.dx, "水平匹配偏移")
        finite(result.dy, "垂直匹配偏移")
        if result.interpolation not in {"nearest", "bilinear"}:
            raise ValueError("插值仅支持最近邻或显式双线性插值")
        if type(result.mirror_x) is not bool or type(result.mirror_y) is not bool:
            raise ValueError("镜像选项必须是布尔值")
        return result


@dataclass(frozen=True)
class PhotometryConfig:
    lut_path: str | None = None
    black_left: float = 0.0
    black_right: float = 0.0
    saturation_code: float | None = None
    normalization: str = "fixed_rms"
    input_semantics: str = "sensor_codes"
    curve: str = "identity"
    input_max: float = 1023.0
    gamma: float = 1.0

    @classmethod
    def from_dict(cls, value):
        result = cls(**(value or {}))
        finite(result.black_left, "左路暗场偏置")
        finite(result.black_right, "右路暗场偏置")
        if result.saturation_code is not None:
            finite(result.saturation_code, "饱和 / 坏点码值阈值")
        if result.lut_path is not None and not isinstance(result.lut_path, str):
            raise ValueError("线性化 LUT 必须为本地文件路径")
        if result.normalization not in {"fixed_rms", "none"}:
            raise ValueError("仅支持全源固定 RMS 尺度或保留校准强度")
        if result.input_semantics not in {"sensor_codes","linear_intensity"}:
            raise ValueError("需明确输入是保存码值还是已确认的线性强度")
        if result.curve not in {"identity", "linear", "sqrt", "square", "gamma", "log"}:
            raise ValueError("未知强度预设曲线")
        if finite(result.input_max, "固定输入尺度") <= 0 or finite(result.gamma, "曲线指数") <= 0:
            raise ValueError("固定输入尺度和曲线指数需大于 0")
        if result.lut_path and result.curve != "identity":
            raise ValueError("自定义 LUT 与内置曲线只选择一种")
        return result


@dataclass(frozen=True)
class CorrelationConfig:
    geometry: GeometryConfig
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    window: int = 101
    stride: int = 100
    lag_min: int = -10
    lag_max: int = 10
    lag_step: int = 1
    lag_list: tuple[int, ...] | None = None
    start: int = 0
    end: int | None = None
    center: int | None = None
    roi: ROI | None = None
    points: tuple[tuple[int, int], ...] = ()
    roi_estimator: str = "pixel_mean"
    min_mean: float = 1e-9
    min_samples: int = 2
    memory_mb: int = 256
    backend: str = "cpu"
    output_dtype: str = "float32"
    workflow: tuple[dict, ...] | None = None
    workflow_version: int = 1

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.get("workflow") or (value.get("workflow_version",1)>=2 and value.get("workflow") is not None):
            from .workflow import resolve_workflow
            resolve_workflow(value)
        else:
            value["workflow"] = None
        value["geometry"] = GeometryConfig.from_dict(value["geometry"])
        value["photometry"] = PhotometryConfig.from_dict(value.get("photometry"))
        if value.get("roi") is not None:
            value["roi"] = ROI(**value["roi"])
        if value.get("lag_list") is not None:
            value["lag_list"] = tuple(value["lag_list"])
        value["points"] = tuple(tuple(p) for p in value.get("points", []))
        result = cls(**value)
        for key, minimum in (("window", 1), ("stride", 1), ("lag_step", 1), ("start", 0), ("min_samples", 1), ("memory_mb", 64)):
            integer(getattr(result, key), key, minimum)
        if result.window % 2 != 1:
            raise ValueError("平均窗口需为奇数，以实际帧作为中心")
        if result.min_samples > result.window:
            raise ValueError("最少有效样本数不能大于窗口")
        if result.memory_mb > 4096:
            raise ValueError("内存预算上限为 4096 MiB")
        for key in ("end", "center"):
            if getattr(result, key) is not None:
                integer(getattr(result, key), key, 0)
        integer(result.lag_min, "延迟下限")
        integer(result.lag_max, "延迟上限")
        if result.lag_min > result.lag_max:
            raise ValueError("延迟下限不能大于上限")
        if finite(result.min_mean, "最小强度均值") <= 0:
            raise ValueError("最小强度均值必须为正数")
        if result.backend not in {"cpu", "auto", "gpu"}:
            raise ValueError("设备仅支持 CPU / Auto / GPU")
        if result.output_dtype not in {"float32","float64"}:
            raise ValueError("系数图精度仅支持 float32 / float64")
        if result.roi_estimator not in {"pixel_mean", "intensity_sum"}:
            raise ValueError("未知 ROI 估计方式")
        if len(result.points) > 32:
            raise ValueError("一次曲线任务最多选择 32 个点")
        for p in result.points:
            if len(p) != 2:
                raise ValueError("点坐标必须为 X、Y")
            for coordinate in p:
                integer(coordinate, "点坐标", 0)
        result.lags()
        return result

    def lags(self):
        if self.lag_list is None:
            if max(abs(self.lag_min),abs(self.lag_max)) > 8_000_000:
                raise ValueError("帧延迟超出支持范围")
            proposed = range(self.lag_min, self.lag_max+1, self.lag_step)
            if len(proposed)>4096:
                raise ValueError("延迟列表最多 4096 个值，请增大延迟扫描步长")
            values = list(proposed)
        else:
            values = list(self.lag_list)
        if not values or len(values) > 4096:
            raise ValueError("延迟列表需包含 1–4096 个值")
        for value in values:
            integer(value, "帧延迟")
            if abs(value) > 8_000_000:
                raise ValueError("帧延迟超出支持范围")
        if self.lag_list is None and self.lag_min <= 0 <= self.lag_max:
            values.append(0)
        values = tuple(sorted(set(values)))
        if len(values)>4096:
            raise ValueError("包含零延迟后的列表超过 4096 个值，请增大延迟扫描步长")
        return values

    def plan(self, count, *, single=False):
        integer(count, "来源帧数", 1)
        a, b = self.start, count if self.end is None else self.end
        if not 0 <= a < b <= count:
            raise ValueError("分析范围需在来源内，且终点不包含")
        h, lags = self.window // 2, self.lags()
        first = a + h + max(0, -min(lags))
        last = b - 1 - h - max(0, max(lags))
        if first > last:
            raise ValueError("范围不足以覆盖完整平均窗口和正负延迟")
        if single:
            if self.center is None or not first <= self.center <= last:
                raise ValueError(f"中心帧需位于索引 {first}～{last}（界面帧号 {first+1}～{last+1}）")
            centers = range(self.center, self.center + 1)
        else:
            centers = range(first, last + 1, self.stride)
        return {"start": a, "end": b, "first_center": centers[0], "last_center": centers[-1], "centers": centers, "lags": lags, "window_count": len(centers), "lag_count": len(lags), "trim_head": h + max(0, -min(lags)), "trim_tail": h + max(0, max(lags))}

    def to_dict(self):
        return asdict(self)
