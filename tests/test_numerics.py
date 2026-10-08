from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from eip.algorithms import BUILTINS, Cancelled, Pipeline, RunContext, load_manifest, validate_parameters
from eip.display import DisplaySettings, labels_to_display, to_display
from eip.domain import ROI, Selection, StepSpec


def test_display_never_changes_scientific_pixels():
    pixels = np.array([[0, 1000], [32000, 65535]], dtype=np.uint16)
    before = pixels.copy()
    rgb = to_display(pixels, DisplaySettings(0, 65535, 2.0, "heat"))
    assert rgb.dtype == np.uint8 and rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(pixels, before)
    assert pixels.dtype == np.uint16


def test_display_handles_nan_without_coercing_input():
    pixels = np.array([[np.nan, np.inf], [-np.inf, 0.5]], dtype=np.float32)
    rgb = to_display(pixels, DisplaySettings(0, 1))
    assert rgb[0, 0, 0] == 0 and rgb[0, 1, 0] == 255
    assert np.isnan(pixels[0, 0]) and np.isposinf(pixels[0, 1])


def test_gain_clips_integer_results_and_preserves_input():
    pixels = np.array([[0, 1000, 40000]], dtype=np.uint16)
    pipeline = Pipeline([StepSpec("intensity", {"gain": 2, "offset": -100})], RunContext())
    try:
        result = pipeline.execute(pixels)
        np.testing.assert_array_equal(result, [[0, 1900, 65535]])
        np.testing.assert_array_equal(pixels, [[0, 1000, 40000]])
        assert result.dtype == np.uint16
    finally:
        pipeline.close()


def test_blur_matches_hand_checkable_reflection():
    pixels = np.arange(9, dtype=np.float32).reshape(3, 3)
    padded = np.pad(pixels, 1, mode="reflect")
    expected = np.array([[padded[y:y + 3, x:x + 3].mean() for x in range(3)] for y in range(3)])
    pipeline = Pipeline([StepSpec("box_blur", {"radius": 1})], RunContext())
    try:
        np.testing.assert_allclose(pipeline.execute(pixels), expected)
    finally:
        pipeline.close()


def test_threshold_is_discrete_label_and_palette_is_stable():
    pipeline = Pipeline([StepSpec("threshold", {"threshold": 1000})], RunContext())
    try:
        result = pipeline.execute(np.array([[999, 1000, 50000]], dtype=np.uint16))
        np.testing.assert_array_equal(result, [[0, 1, 1]])
        assert result.dtype == np.uint8 and pipeline.output_kind == "label"
        rgb = labels_to_display(result)
        assert not rgb[0, 0].any()
        np.testing.assert_array_equal(rgb[0, 1], rgb[0, 2])
    finally:
        pipeline.close()


@pytest.mark.parametrize("values", [{"gain": float("nan")}, {"gain": True}, {"extra": 1}])
def test_invalid_parameters_are_rejected(values):
    with pytest.raises(ValueError):
        validate_parameters(BUILTINS[0], values)


def test_roi_uses_native_coordinates_and_rejects_invalid_bounds():
    pixels = np.arange(30).reshape(5, 6)
    np.testing.assert_array_equal(ROI(2, 1, 3, 2).crop(pixels), [[8, 9, 10], [14, 15, 16]])
    with pytest.raises(ValueError):
        ROI(5, 0, 2, 1).crop(pixels)
    with pytest.raises(ValueError):
        Selection("range", 3, 3)


def test_external_plugin_and_changed_code_detection(tmp_path):
    manifest = Path(__file__).resolve().parents[1] / "examples/plugins/invert.json"
    spec = load_manifest(str(manifest))
    step = StepSpec(spec.id, {"reference": 65535}, manifest=spec.manifest, code_hash=spec.code_hash)
    pipeline = Pipeline([step], RunContext())
    try:
        result = pipeline.execute(np.array([[0, 1000, 65535]], dtype=np.uint16))
        np.testing.assert_array_equal(result, [[65535, 64535, 0]])
        assert result.dtype == np.float32
    finally:
        pipeline.close()
    with pytest.raises(ValueError, match="已变化"):
        Pipeline([replace(step, code_hash="stale")], RunContext())


def test_cancellation_checked_before_work():
    with pytest.raises(Cancelled):
        Pipeline([StepSpec("intensity")], RunContext(lambda: True))
