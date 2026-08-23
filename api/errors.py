"""Lightweight public exception types shared by app and registry."""

from __future__ import annotations


class InferenceUnavailableError(RuntimeError):
    pass


class InvalidRequestError(ValueError):
    def __init__(self, code: str, message: str, details: dict[str, object] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details


class MatchNotFoundError(LookupError):
    def __init__(self, match_id: int):
        super().__init__(f"Match {match_id} was not found in the frozen feature store.")
        self.match_id = match_id


class SnapshotNotFoundError(LookupError):
    def __init__(self, match_id: int, minute: int):
        super().__init__(f"Snapshot minute {minute} is unavailable for match {match_id}.")
        self.match_id = match_id
        self.minute = minute
