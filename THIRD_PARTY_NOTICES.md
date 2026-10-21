# Third-Party Notices

## Ultralytics YOLOv8

- Version: 8.4.174
- License: AGPL-3.0 (https://ultralytics.com/license)
- Usage: training, export and inference framework. The derived checkpoint
  `artifacts/models/pcba_yolov8n_best.pt` and ONNX export
  `artifacts/models/pcba_yolov8n.onnx` inherit AGPL-3.0 terms from the
  framework/checkpoint metadata. Commercial or closed-source use requires
  compliance with AGPL-3.0 or an Ultralytics enterprise license.

## PCBA dataset (`data_y.zip`)

- Status: **internal / proprietary data**.
- Not redistributed in this repository, and no redistribution authorization
  is granted here. Users must obtain permission from the data owner before
  using or redistributing the archive.
- The repository's MIT license applies to the source code only and does not
  grant rights to the dataset or model weights.

## Other dependencies

See `pyproject.toml` for the full dependency set. Each package retains its own
license; notably PyTorch (BSD-style), ONNX/ONNX Runtime (MIT/Apache-2.0),
Gradio (Apache-2.0), OpenCV (Apache-2.0), and Pillow (HPND).
