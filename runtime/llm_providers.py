"""Brain (LLM) provider catalog for the voice exam.

Every provider here exposes an OpenAI-compatible streaming chat endpoint, so
one client library (``openai``) serves all of them and the fallback layer in
``llm_resilience`` never has to know which vendor it is talking to.
"""

from dataclasses import dataclass

from openai import NOT_GIVEN

from runtime.settings import RuntimeSettings

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class LLMCandidate:
    provider: str
    model: str
    api_key: str
    base_url: str | None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def provider_for_model(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if lowered.startswith("claude-"):
        return "anthropic"
    if lowered.startswith("gemini-"):
        return "gemini"
    return "mistral"


def _provider_config(provider: str, settings: RuntimeSettings) -> tuple[str, str | None]:
    if provider == "openai":
        return settings.openai_api_key, None
    if provider == "anthropic":
        return settings.anthropic_api_key, ANTHROPIC_BASE_URL
    if provider == "gemini":
        return settings.gemini_api_key, GEMINI_BASE_URL
    return settings.mistral_api_key, MISTRAL_BASE_URL


def resolve_candidate(model: str, settings: RuntimeSettings) -> LLMCandidate | None:
    """Return a usable candidate for ``model`` or None if its provider has no key."""
    provider = provider_for_model(model)
    api_key, base_url = _provider_config(provider, settings)
    if not api_key:
        return None
    return LLMCandidate(provider=provider, model=model, api_key=api_key, base_url=base_url)


def build_candidates(
    primary_model: str, fallback_models: list[str], settings: RuntimeSettings
) -> tuple[list[LLMCandidate], list[str]]:
    """Primary first, then fallbacks in order; duplicates and keyless models dropped.

    Returns the usable candidates plus the model ids that were skipped for
    lack of an API key, so the caller can log them once per session.
    """
    candidates: list[LLMCandidate] = []
    skipped: list[str] = []
    for model in [primary_model, *fallback_models]:
        if not model or any(c.model == model for c in candidates):
            continue
        candidate = resolve_candidate(model, settings)
        if candidate is None:
            skipped.append(model)
        else:
            candidates.append(candidate)
    return candidates, skipped


def shape_completion_params(candidate: LLMCandidate, params: dict) -> dict:
    """Adjust OpenAI-style params for providers/models that reject some of them.

    GPT-5.x reasoning models only accept ``max_completion_tokens`` (not
    ``max_tokens``) and the default temperature; ``reasoning_effort="low"``
    keeps their TTFB in the same band as the non-reasoning providers.
    """
    if candidate.provider != "openai":
        return params
    shaped = dict(params)
    max_tokens = shaped.pop("max_tokens", NOT_GIVEN)
    if max_tokens is not NOT_GIVEN and max_tokens is not None:
        shaped["max_completion_tokens"] = max_tokens
    shaped["temperature"] = NOT_GIVEN
    shaped["reasoning_effort"] = "low"
    return shaped
