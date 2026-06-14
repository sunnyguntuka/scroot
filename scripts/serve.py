"""Quick launcher: python serve.py"""
from scroot.dashboard.server import create_app
import uvicorn

app = create_app(store_path="./scroot_store.jsonl")
print("  Scroot Review Console -> http://127.0.0.1:7433")
uvicorn.run(app, host="127.0.0.1", port=7433, log_level="warning")
