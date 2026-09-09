"""Bounded evidence metadata. References are never fetched or treated as content."""

from typing import Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: Optional[str] = Field(default=None, max_length=200)
    role: Literal["user", "assistant", "tool"] = "user"
    modality: Literal["text", "image", "audio", "document", "link", "mixed"] = "text"
    representation: Literal["text", "extracted_text", "reference_only"] = "text"
    assertion: Literal["reported", "derived", "inferred"] = "reported"
    delivery: Literal["received", "generated", "delivered", "unknown"] = "unknown"
    observed_at: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    reference: Optional[str] = Field(default=None, max_length=1000)
    confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @field_validator("reference")
    @classmethod
    def safe_reference(cls, value):
        if not value:
            return None
        parts = urlsplit(value)
        if parts.scheme in {"http", "https"}:
            # Signed query strings and userinfo are neither evidence nor durable references.
            if not parts.hostname or parts.username or parts.password:
                raise ValueError("reference must not contain credentials")
            return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        # Local filenames/paths are host-owned metadata; never dereference them.
        if parts.scheme or "\n" in value or "\r" in value:
            raise ValueError("unsupported reference")
        return value


def content_text(content):
    """Consume existing text blocks only; ignore binary blocks and unsupported shapes."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part["text"] for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        and isinstance(part.get("text"), str)
    ).strip()


class MemoryValidity(BaseModel):
    """Validate imported intervals before any row is written."""

    valid_from: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    valid_to: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    supersedes: Optional[str] = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def ordered(self):
        if self.valid_from is not None and self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to precedes valid_from")
        return self
