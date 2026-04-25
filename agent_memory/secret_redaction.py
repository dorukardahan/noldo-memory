"""Secret-like value detection and redaction helpers.

These helpers are intentionally conservative about output: callers can count
matches or replace values, but they never need to print the original secret.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable


Replacement = str | Callable[[re.Match[str]], str]


@dataclass(frozen=True)
class SecretPattern:
    name: str
    pattern: re.Pattern[str]
    replacement: Replacement

    def redact(self, text: str) -> tuple[str, int]:
        return self.pattern.subn(self.replacement, text)


def _assignment_replacement(match: re.Match[str]) -> str:
    return f"{match.group('label')}{match.group('sep')}<redacted:{match.group('kind')}>"


SECRET_PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern(
        "private_key_block",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "<redacted:private_key_block>",
    ),
    SecretPattern(
        "bearer_token",
        re.compile(r"\bBearer\s+(?!<redacted[:>])[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
        "Bearer <redacted:bearer_token>",
    ),
    SecretPattern(
        "slack_token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}|\bxapp-[A-Za-z0-9-]{12,}"),
        "<redacted:slack_token>",
    ),
    SecretPattern(
        "openai_like_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
        "<redacted:openai_like_key>",
    ),
    SecretPattern(
        "api_key_assignment",
        re.compile(
            r"\b(?P<label>(?P<kind>api[_-]?key|token|secret|password|passwd|pwd))"
            r"(?P<sep>\s*[:=]\s*)"
            r"(?P<value>(?!<redacted[:>])(?!(?:['\"])?<redacted)[^\s,;]+)",
            re.IGNORECASE,
        ),
        _assignment_replacement,
    ),
)


def count_secret_like(text: str | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not text:
        return counts
    for secret_pattern in SECRET_PATTERNS:
        matches = secret_pattern.pattern.findall(text)
        if matches:
            counts[secret_pattern.name] += len(matches)
    return counts


def redact_secret_like(text: str | None) -> tuple[str | None, Counter[str]]:
    if text is None:
        return None, Counter()

    redacted = text
    counts: Counter[str] = Counter()
    for secret_pattern in SECRET_PATTERNS:
        redacted, match_count = secret_pattern.redact(redacted)
        if match_count:
            counts[secret_pattern.name] += match_count
    return redacted, counts
