from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BackendError(Exception):
    code: str
    message: str
    action: str | None = None

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.action:
            payload["action"] = self.action
        return payload
