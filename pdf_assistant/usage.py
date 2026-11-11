"""
Token usage and estimated cost of OpenAI calls.

Wrap work in `with usage.track() as tracked:`; every API call made
inside it (in the same thread) is recorded in `tracked`. Outside a
`track()` block, recording does nothing.
"""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import config


def cost_of(model, input_tokens, output_tokens=0):
    """Estimated cost in US$; 0 for models missing from the price list."""

    input_price, output_price = config.PRICES_PER_MILLION.get(model, (0.0, 0.0))

    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@dataclass
class Call:
    purpose: str  # search, rewrite, rerank, answer, indexing
    model: str
    input_tokens: int
    output_tokens: int = 0

    @property
    def cost(self):
        return cost_of(self.model, self.input_tokens, self.output_tokens)


@dataclass
class Usage:
    calls: list = field(default_factory=list)

    @property
    def input_tokens(self):
        return sum(call.input_tokens for call in self.calls)

    @property
    def output_tokens(self):
        return sum(call.output_tokens for call in self.calls)

    @property
    def cost(self):
        return sum(call.cost for call in self.calls)

    def to_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "calls": [
                {
                    "purpose": call.purpose,
                    "model": call.model,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "cost": call.cost,
                }
                for call in self.calls
            ],
        }


_current = contextvars.ContextVar("usage", default=None)


@contextmanager
def track():
    usage = Usage()
    token = _current.set(usage)

    try:
        yield usage
    finally:
        _current.reset(token)


def record(purpose, model, input_tokens, output_tokens=0):
    usage = _current.get()

    if usage is not None:
        usage.calls.append(
            Call(purpose, model, int(input_tokens or 0), int(output_tokens or 0))
        )


def record_response(purpose, model, response_usage):
    """Record the `usage` object of a Responses API reply (may be None)."""

    if response_usage is not None:
        record(
            purpose,
            model,
            getattr(response_usage, "input_tokens", 0),
            getattr(response_usage, "output_tokens", 0),
        )


def record_embedding(purpose, response):
    """Record the token count of an Embeddings API reply."""

    response_usage = getattr(response, "usage", None)

    if response_usage is not None:
        record(purpose, config.EMBEDDING_MODEL, getattr(response_usage, "prompt_tokens", 0))


def format_cost(cost):
    if cost == 0:
        return "$0"
    if cost < 0.0001:
        return "<$0.0001"
    return f"~${cost:.4f}"


def format_usage(data):
    """'1,234 tokens · ~$0.0008' from a Usage.to_dict() result."""

    tokens = data["input_tokens"] + data["output_tokens"]

    return f"{tokens:,} tokens · {format_cost(data['cost'])}"


def total(usages):
    """Add up several Usage.to_dict() results."""

    usages = [data for data in usages if data]

    return {
        "input_tokens": sum(data["input_tokens"] for data in usages),
        "output_tokens": sum(data["output_tokens"] for data in usages),
        "cost": sum(data["cost"] for data in usages),
    }
