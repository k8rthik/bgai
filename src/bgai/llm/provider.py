"""Provider-agnostic LLM interface (Phase 7).

The master plan requires the LLM track to run against both the Claude
API and the university's Azure OpenAI access, so nothing above this
module may import a vendor SDK. A provider takes messages and returns
text; that is the whole contract.

``MockProvider`` exists so the entire ladder -- prompt construction,
tool loop, answer parsing, arena integration -- is testable and
CI-runnable with no API key and no spend. Every rung's *plumbing* is
verified offline; only the *measurement* of a rung needs credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Usage:
    """Token accounting -- the master plan requires per-move token
    budgets to be logged, since expensive rungs must play fewer games."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


class Provider(Protocol):
    name: str

    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str: ...

    @property
    def usage(self) -> Usage: ...


@dataclass
class MockProvider:
    """Scripted provider for tests: returns canned replies in order, or
    delegates to a callable that sees the messages.
    """

    replies: Sequence[str] = field(default_factory=tuple)
    responder: Callable[[Sequence[Message]], str] | None = None
    name: str = "mock"
    _index: int = 0
    _usage: Usage = field(default_factory=Usage)

    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        self._usage.calls += 1
        self._usage.input_tokens += sum(len(m.content) // 4 for m in messages)
        if self.responder is not None:
            out = self.responder(messages)
        elif self.replies:
            out = self.replies[self._index % len(self.replies)]
            self._index += 1
        else:
            out = "0"
        self._usage.output_tokens += len(out) // 4
        return out

    @property
    def usage(self) -> Usage:
        return self._usage


@dataclass
class AnthropicProvider:
    """Claude API adapter. Imports the SDK lazily so the package stays
    importable (and testable) without the dependency or a key.
    """

    model: str = "claude-sonnet-5"
    name: str = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"
    _client: object | None = None
    _usage: Usage = field(default_factory=Usage)

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "anthropic SDK not installed; `uv add anthropic` to use this provider"
                ) from exc
            if not os.environ.get(self.api_key_env):
                raise RuntimeError(
                    f"{self.api_key_env} is not set -- the LLM track cannot run without it"
                )
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        client = self._ensure_client()
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        response = client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=max_tokens,
            system=system or None,
            messages=turns,
        )
        self._usage.calls += 1
        self._usage.input_tokens += response.usage.input_tokens
        self._usage.output_tokens += response.usage.output_tokens
        return "".join(block.text for block in response.content if block.type == "text")

    @property
    def usage(self) -> Usage:
        return self._usage
