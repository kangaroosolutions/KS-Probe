"""
AdapterRegistry — maps model IDs to adapter instances.

Usage:
    registry = AdapterRegistry()
    adapter = registry.get("claude-sonnet-4-6")
    response = await adapter.send_prompt(...)
"""
from __future__ import annotations

from typing import Dict, Optional, Type

from ks_probe.adapters.base import BaseModelAdapter

# Provider → adapter class (imported lazily to avoid hard dependencies)
_PROVIDER_MAP: Dict[str, str] = {
    "openai": "ks_probe.adapters.openai_adapter.OpenAIAdapter",
    "anthropic": "ks_probe.adapters.anthropic_adapter.AnthropicAdapter",
    "google": "ks_probe.adapters.google_adapter.GoogleAdapter",
    "xai": "ks_probe.adapters.xai_adapter.XAIAdapter",
    "deepseek": "ks_probe.adapters.deepseek_adapter.DeepSeekAdapter",
    "perplexity": "ks_probe.adapters.perplexity_adapter.PerplexityAdapter",
    "vllm": "ks_probe.adapters.vllm_adapter.VLLMAdapter",
    "mock": "ks_probe.adapters.mock_adapter.MockAdapter",
}

# Model ID → provider
_MODEL_PROVIDER: Dict[str, str] = {
    "gpt-5.2": "openai",
    "gpt-4.1": "openai",
    "claude-opus-4-6": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "gemini-3.1-pro": "google",
    "gemini-2.5-pro": "google",
    "grok-4.1-fast": "xai",
    "deepseek-v3.2": "deepseek",
    "deepseek-r1": "deepseek",
    "sonar-pro": "perplexity",
    "sonar": "perplexity",
    "sonar-reasoning-pro": "perplexity",
    "llama-4-maverick": "vllm",
    "llama-4-scout": "vllm",
    "mock": "mock",
}


def _import_class(dotted_path: str) -> Type[BaseModelAdapter]:
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class AdapterRegistry:
    def __init__(self) -> None:
        self._cache: Dict[str, BaseModelAdapter] = {}

    def get(self, model_id: str, **kwargs) -> BaseModelAdapter:
        """Return a (cached) adapter instance for the given model ID."""
        if model_id in self._cache:
            return self._cache[model_id]

        provider = _MODEL_PROVIDER.get(model_id)
        if provider is None:
            raise ValueError(
                f"Unknown model '{model_id}'. "
                f"Known models: {list(_MODEL_PROVIDER.keys())}"
            )

        cls_path = _PROVIDER_MAP[provider]
        AdapterClass = _import_class(cls_path)
        adapter = AdapterClass(model_id=model_id, **kwargs)
        self._cache[model_id] = adapter
        return adapter

    def register(self, model_id: str, adapter: BaseModelAdapter) -> None:
        """Register a custom adapter instance."""
        self._cache[model_id] = adapter

    @property
    def known_models(self):
        return list(_MODEL_PROVIDER.keys())


# Module-level singleton
_registry: Optional[AdapterRegistry] = None


def get_registry() -> AdapterRegistry:
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry
