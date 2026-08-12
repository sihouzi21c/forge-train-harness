"""Compatibility shims — single import point for optional stdlib modules."""

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = ["tomllib"]
