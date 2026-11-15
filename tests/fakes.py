"""Offline stand-ins for Gemini so the chain logic can be tested for free."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk

from src.config import Settings


def make_settings(**overrides) -> Settings:
    base = {
        "api_key": "test-key",
        "model": "gemini-test",
        "chunk_size_tokens": 100,
        "chunk_overlap_tokens": 20,
        "single_pass_token_limit": 100_000,
        "temperature": 0.2,
        "request_timeout_seconds": 30,
    }
    base.update(overrides)
    return Settings(**base)


class FakeLLM:
    """Duck-typed replacement for ChatGoogleGenerativeAI.

    Only `invoke` and `stream` are used by src.chains, so a small stub keeps the
    tests fast and free of provider coupling.
    """

    def __init__(
        self,
        reply: str = "## TL;DR\nA short summary.\n\n## Key Points\n- one [p.1]",
        usage: tuple[int, int] = (100, 20),
        fail_with: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.usage = usage
        self.fail_with = fail_with
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail_with:
            raise self.fail_with
        return AIMessage(
            content=self.reply,
            usage_metadata={
                "input_tokens": self.usage[0],
                "output_tokens": self.usage[1],
                "total_tokens": sum(self.usage),
            },
        )

    def stream(self, messages):
        self.calls.append(messages)
        if self.fail_with:
            raise self.fail_with
        for line in self.reply.splitlines(keepends=True):
            yield AIMessageChunk(content=line)
