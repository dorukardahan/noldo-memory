from __future__ import annotations

import builtins
import importlib
import sys


def test_reranker_module_import_is_lazy(monkeypatch):
    sys.modules.pop("agent_memory.reranker", None)

    attempted = []
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in {"sentence_transformers", "torch"}:
            attempted.append(name)
            raise AssertionError(f"unexpected eager import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    module = importlib.import_module("agent_memory.reranker")

    assert attempted == []
    assert module.CrossEncoderReranker(enabled=False).available is False
    assert attempted == []
