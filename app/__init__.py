"""
Retriva application package.

This module runs before any submodule is imported, so it is the right place to
force HuggingFace libraries into offline mode. All required models live in the
local cache, and this guarantees the app never calls out to the network.
"""

import os

if os.getenv("HF_LOCAL_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
