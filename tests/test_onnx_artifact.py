from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnxruntime = pytest.importorskip("onnxruntime")

MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "pcba_yolov8n.onnx"


@pytest.mark.skipif(not MODEL.exists(), reason="发布版 ONNX 不存在")
def test_published_onnx_loads_and_runs() -> None:
    session = onnxruntime.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    assert inputs[0].name == "images"
    assert inputs[0].shape == [1, 3, 960, 960]

    output = session.run(None, {"images": np.zeros((1, 3, 960, 960), dtype=np.float32)})[0]
    assert output.shape == (1, 9, 18900)
    assert np.isfinite(output).all()
