"""pytest 設定 — 依存パッケージがない場合のテストスキップ."""
import importlib.util
from pathlib import Path

# uvicorn がない環境では test_inference を収集対象から除外
collect_ignore = []
if importlib.util.find_spec("uvicorn") is None:
    collect_ignore.append(str(Path(__file__).parent / "test_inference.py"))
