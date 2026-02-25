"""Tests for token_utils module."""

from agent_memory.token_utils import estimate_tokens, trim_results_to_budget


def test_estimate_tokens_basic():
    assert estimate_tokens("hello") >= 1
    assert estimate_tokens("a" * 100) == 25  # 100 / 4.0


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 1  # min 1


def test_trim_results_empty():
    assert trim_results_to_budget([], 1000) == []


def test_trim_results_all_fit():
    results = [
        {"id": "1", "text": "short text"},
        {"id": "2", "text": "another short"},
    ]
    trimmed = trim_results_to_budget(results, 1000)
    assert len(trimmed) == 2


def test_trim_results_budget_exceeded():
    results = [
        {"id": "1", "text": "a" * 400},  # ~100 tokens
        {"id": "2", "text": "b" * 400},  # ~100 tokens
        {"id": "3", "text": "c" * 400},  # ~100 tokens
    ]
    trimmed = trim_results_to_budget(results, 150)
    assert len(trimmed) == 1  # only first fits


def test_trim_results_at_least_one():
    """Even if first result exceeds budget, include it."""
    results = [
        {"id": "1", "text": "a" * 1000},  # ~250 tokens
    ]
    trimmed = trim_results_to_budget(results, 100)
    assert len(trimmed) == 1  # at least one


def test_trim_results_preserves_order():
    results = [
        {"id": "1", "text": "first"},
        {"id": "2", "text": "second"},
        {"id": "3", "text": "third"},
    ]
    trimmed = trim_results_to_budget(results, 1000)
    assert [r["id"] for r in trimmed] == ["1", "2", "3"]
