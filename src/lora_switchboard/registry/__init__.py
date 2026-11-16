"""SQLite-backed registry of adapters, versions, aliases and rollout events."""

from .repository import Registry

__all__ = ["Registry"]
