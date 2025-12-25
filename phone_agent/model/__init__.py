"""Model client module for AI inference."""

from phone_agent.model.client import ModelClient, ModelConfig

try:
    from phone_agent.model.client_ollama import OllamaModelClient, OllamaModelConfig
    __all__ = ["ModelClient", "ModelConfig", "OllamaModelClient", "OllamaModelConfig"]
except ImportError:
    # Ollama-specific modules may not be available
    __all__ = ["ModelClient", "ModelConfig"]
