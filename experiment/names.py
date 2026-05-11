"""Shared naming helpers for release output folders and files."""

from __future__ import annotations

from typing import List


def slugify(text: object, default: str = "item") -> str:
    """Turn labels into stable folder/file tokens used throughout the release."""
    out: List[str] = []
    for char in str(text).strip():
        if char.isalnum() or char in "_.-":
            out.append(char)
        elif char.isspace():
            out.append("_")
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or default
