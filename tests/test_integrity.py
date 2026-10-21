from __future__ import annotations

import json

import pytest

from pcba_defect.cli import _build_parser
from pcba_defect.constants import CLASS_NAMES, PROJECT_ROOT


def test_cli_has_required_commands() -> None:
    parser = _build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert set(subparsers.choices) == {"prepare", "analyze", "train", "eval", "predict", "export", "app"}


def test_prepared_dataset_integrity_when_available() -> None:
    summary_path = PROJECT_ROOT / "data" / "pcba_yolo" / "dataset_summary.json"
    if not summary_path.exists():
        pytest.skip("本地未准备数据集（data/pcba_yolo/dataset_summary.json 不存在）")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["totals"]["images"] == 600
    assert summary["totals"]["objects"] == 4552
    assert set(summary["class_mapping"]) == set(CLASS_NAMES)
    assert summary["splits"]["train"]["images"] == 350
    assert summary["splits"]["val"]["images"] == 100
    assert summary["splits"]["test"]["images"] == 150


def test_find_best_model_discovers_published_artifact(monkeypatch, tmp_path) -> None:
    import pcba_defect.utils as utils

    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)
    model = tmp_path / "artifacts" / "models" / "pcba_yolov8n_best.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"stub")
    assert utils.find_best_model() == model.resolve()
