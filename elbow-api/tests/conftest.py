import importlib.util
import sys
import os

# テスト実行時にAPIディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# fastapi / uvicorn がない環境ではこのディレクトリ全体をスキップ
collect_ignore_glob = []
if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None:
    collect_ignore_glob.append("*.py")
