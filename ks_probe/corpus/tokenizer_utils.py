"""
Per-model tokenizer wrappers.

Returns a callable count_fn(text) → int for any supported model.
Tokenizers are loaded lazily and cached.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Optional


def _tiktoken_counter(encoding_name: str) -> Callable[[str], int]:
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    return lambda text: len(enc.encode(text))


def _hf_counter(model_name: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def _char_ratio_counter(chars_per_token: float) -> Callable[[str], int]:
    return lambda text: max(1, int(len(text) / chars_per_token))


# Model ID → (loader_fn, args)
_TOKENIZER_REGISTRY: Dict[str, Callable[[], Callable[[str], int]]] = {
    "gpt-5.2":           lambda: _tiktoken_counter("cl100k_base"),
    "gpt-4.1":           lambda: _tiktoken_counter("cl100k_base"),
    "claude-opus-4-6":   lambda: _char_ratio_counter(3.8),
    "claude-sonnet-4-6": lambda: _char_ratio_counter(3.8),
    "gemini-3.1-pro":    lambda: _char_ratio_counter(3.7),
    "gemini-2.5-pro":    lambda: _char_ratio_counter(3.7),
    "grok-4.1-fast":     lambda: _tiktoken_counter("cl100k_base"),
    "deepseek-v3.2":     lambda: _tiktoken_counter("cl100k_base"),
    "deepseek-r1":       lambda: _tiktoken_counter("cl100k_base"),
    "llama-4-maverick":  lambda: _char_ratio_counter(3.9),
    "llama-4-scout":     lambda: _char_ratio_counter(3.9),
    "mock":              lambda: _char_ratio_counter(4.0),
}

_cache: Dict[str, Callable[[str], int]] = {}


def get_tokenizer(model_id: str) -> Callable[[str], int]:
    """Return a cached token-counting function for the given model."""
    if model_id not in _cache:
        if model_id not in _TOKENIZER_REGISTRY:
            # Fallback: generic word-based estimate
            _cache[model_id] = lambda text: max(1, int(len(text.split()) * 1.3))
        else:
            try:
                _cache[model_id] = _TOKENIZER_REGISTRY[model_id]()
            except Exception:
                # If tokenizer library not installed, fall back to char ratio
                _cache[model_id] = _char_ratio_counter(4.0)
    return _cache[model_id]


def count_tokens(text: str, model_id: str) -> int:
    return get_tokenizer(model_id)(text)


def get_all_model_ids() -> list:
    return list(_TOKENIZER_REGISTRY.keys())
