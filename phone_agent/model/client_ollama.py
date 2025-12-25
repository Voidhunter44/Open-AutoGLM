"""Ollama-specific model client for AI inference using OpenAI-compatible API."""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from phone_agent.config.i18n import get_message
from phone_agent.model.client import ModelClient, ModelConfig, ModelResponse


@dataclass
class OllamaModelConfig(ModelConfig):
    """Configuration for Ollama-based AI model."""

    base_url: str = "http://localhost:11434/v1"  # Ollama default endpoint
    api_key: str = "EMPTY"  # Ollama doesn't typically require API keys
    model_name: str = "qwen3-vl:4b"  # Default Ollama model for this integration
    max_tokens: int = 3000
    temperature: float = 0.0
    top_p: float = 0.85
    frequency_penalty: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)
    lang: str = "cn"  # Language for UI messages: 'cn' or 'en'


class OllamaModelClient(ModelClient):
    """
    Ollama-specific client for interacting with Ollama vision-language models.

    This client handles potential differences in response format from Ollama models,
    particularly Qwen models, while maintaining compatibility with the base ModelClient interface.

    Args:
        config: Ollama model configuration.
    """

    def __init__(self, config: OllamaModelConfig | None = None):
        # Ensure we're using Ollama-specific configuration
        if config is None:
            config = OllamaModelConfig()
        elif not isinstance(config, OllamaModelConfig):
            # Convert to OllamaModelConfig if needed, preserving values
            config = OllamaModelConfig(
                base_url=config.base_url,
                api_key=config.api_key,
                model_name=config.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                extra_body=config.extra_body,
                lang=config.lang,
            )
        
        super().__init__(config)
        self.config = config
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def _parse_response(self, content: str) -> tuple[str, str]:
        """
        Parse the model response into thinking and action parts, optimized for Qwen/Ollama models.

        Parsing rules:
        1. If content contains '<thinking>' and '</thinking>', extract thinking from between these tags
        2. If content contains 'finish(message=', everything before is thinking,
           everything from 'finish(message=' onwards is action.
        3. If rule 2 doesn't apply but content contains 'do(action=',
           everything before is thinking, everything from 'do(action=' onwards is action.
        4. Fallback: If content contains '<answer>', use legacy parsing with XML tags.
        5. Otherwise, return empty thinking and full content as action.

        Args:
            content: Raw response content.

        Returns:
            Tuple of (thinking, action).
        """
        # Rule 1: Check for <thinking> tags (specifically for Qwen/Ollama models)
        if "<thinking>" in content and "</thinking>" in content:
            try:
                start_idx = content.find("<thinking>") + len("<thinking>")
                end_idx = content.find("</thinking>")
                thinking = content[start_idx:end_idx].strip()
                
                # Look for action after </thinking>
                remaining_content = content[end_idx+len("</thinking>"):].strip()
                
                if "finish(message=" in remaining_content:
                    parts = remaining_content.split("finish(message=", 1)
                    action = "finish(message=" + parts[1] if len(parts) > 1 else remaining_content
                elif "do(action=" in remaining_content:
                    parts = remaining_content.split("do(action=", 1)
                    action = "do(action=" + parts[1] if len(parts) > 1 else remaining_content
                else:
                    action = remaining_content
                
                return thinking, action
            except:
                # If parsing fails, fall back to other methods
                pass

        # Rule 2: Check for finish(message=
        if "finish(message=" in content:
            parts = content.split("finish(message=", 1)
            thinking = parts[0].strip()
            action = "finish(message=" + parts[1]
            return thinking, action

        # Rule 3: Check for do(action=
        if "do(action=" in content:
            parts = content.split("do(action=", 1)
            thinking = parts[0].strip()
            action = "do(action=" + parts[1]
            return thinking, action

        # Rule 4: Fallback to legacy XML tag parsing
        if "<answer>" in content:
            parts = content.split("<answer>", 1)
            thinking = parts[0].replace("[", "").replace("]", "").strip()
            action = parts[1].replace("</answer>", "").strip()
            return thinking, action

        # Rule 5: No markers found, return content as action
        return "", content