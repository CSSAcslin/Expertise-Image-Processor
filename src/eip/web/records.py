"""Human-readable names and reproducible job summaries."""
from __future__ import annotations

from datetime import datetime


def display_name(value, fallback=None):
    if fallback is not None and (value is None or value == ""):
        return fallback
    if not isinstance(value, str):
        raise ValueError("结果名称需为文本")
    name = value.strip()
    if not name or len(name) > 160 or any(ord(c) < 32 for c in name):
        raise ValueError("结果名称需为 1～160 个可显示字符")
    return name


def split_summary(config, task):
    roi = config.roi
    area = f"ROI ({roi.x},{roi.y}) {roi.width}×{roi.height}" if roi else "全幅"
    lags = config.lags()
    delay = f"q={lags[0]:+d}" if len(lags) == 1 else f"q={lags[0]:+d}…{lags[-1]:+d} · {len(lags)} 个延迟"
    center = f"中心帧 {(config.center or 0)+1}" if task in {"trial", "curve"} else f"步长 {config.stride}"
    if task == "statistics":
        return f"全部输入帧 · {config.photometry.curve} · {config.photometry.normalization}"
    return f"W={config.window} · {delay} · {center} · {area}"


def automatic_name(source, kind, created=None):
    moment = datetime.now() if created is None else datetime.fromtimestamp(created)
    return f"{source} · {kind} · {moment.strftime('%m-%d %H:%M:%S')}"


def saved_split_summary(value, task):
    """Historical records remain readable even when today's limits are stricter."""
    from eip.modes.split_correlation.config import CorrelationConfig
    try:
        return split_summary(CorrelationConfig.from_dict(value),task)
    except (ValueError,TypeError,KeyError):
        return f"W={value.get('window','?')} · q={value.get('lag_min','?')}…{value.get('lag_max','?')} · 历史参数（新任务需重新检查）"
