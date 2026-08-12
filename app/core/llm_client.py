"""
llm_client.py — Provider-agnostic LLM wrapper.

Default provider: Anthropic Claude (via `anthropic` SDK).
Model is read from AI_MODEL_NAME in .env (default: claude-sonnet-4-6).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To swap providers (e.g. OpenAI):
  1. Edit only this file.
  2. Change the __init__ to initialise the OpenAI client.
  3. Keep the same public interface: .chat(messages, system, max_tokens) → str
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.config import AI_MODEL_NAME, ANTHROPIC_API_KEY


class LLMClient:
    """
    Thin wrapper over an LLM provider.
    Public interface:
        client.is_available → bool
        client.chat(messages, system, max_tokens) → str
    """

    def __init__(self) -> None:
        self._client = None
        self._available = False

        if not ANTHROPIC_API_KEY:
            print("⚠️  ANTHROPIC_API_KEY not set — chatbot will be unavailable.")
            return

        try:
            import anthropic  # noqa: PLC0415

            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self._available = True
            print(f"✅ LLM client initialised (model: {AI_MODEL_NAME})")
        except ImportError:
            print("⚠️  `anthropic` package not installed. Run: pip install anthropic")

    @property
    def is_available(self) -> bool:
        return self._available

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a chat request and return the assistant's reply as a string.

        Parameters
        ----------
        messages    : list of {"role": "user"|"assistant", "content": str}
        system      : optional system prompt string
        max_tokens  : maximum tokens in the response
        """
        if not self._available:
            return (
                "⚠️ **LLM not configured.** "
                "Please set `ANTHROPIC_API_KEY` in your `.env` file to enable the chatbot."
            )

        kwargs: Dict = {
            "model":      AI_MODEL_NAME,
            "max_tokens": max_tokens,
            "messages":   messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return (or create) the module-level LLMClient singleton."""
    global _instance
    if _instance is None:
        _instance = LLMClient()
    return _instance
