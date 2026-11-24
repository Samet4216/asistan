"""Unit tests for Ollama Model Runtime (Phase 1)."""

import pytest
from pathlib import Path
from personal_ai.runtime.ollama_client import OllamaClient, OllamaRuntimeError


def test_ollama_client_config_loading(tmp_path: Path):
    """Verify configuration parameters are accurately parsed from YAML."""
    custom_yaml = tmp_path / "model.yaml"
    custom_yaml.write_text(
        """
model:
  ollama_base_url: "http://127.0.0.1:11434"
  default_model: "qwen2.5:7b-instruct-q4_K_M"
  temperature: 0.70
  num_ctx: 4096
""",
        encoding="utf-8"
    )

    client = OllamaClient(config_path=str(custom_yaml))
    assert client.base_url == "http://127.0.0.1:11434"
    assert client.default_model == "qwen2.5:7b-instruct-q4_K_M"
    assert client.default_temperature == 0.70
    assert client.default_num_ctx == 4096


def test_ollama_client_fallback_defaults():
    """Verify default fallback parameters when config file is absent."""
    client = OllamaClient(config_path="non_existent_config.yaml")
    # In our implementation, default fallback is 127.0.0.1:11434 to avoid Windows IPv6 resolution issues
    assert client.base_url == "http://127.0.0.1:11434"
    assert client.default_temperature == 0.65
    assert client.default_num_ctx == 8192


def test_ollama_client_unreachable_endpoint():
    """Verify descriptive OllamaRuntimeError when host is unreachable."""
    client = OllamaClient(config_path="non_existent_config.yaml")
    client.base_url = "http://127.0.0.1:9999"  # invalid port

    assert client.is_alive() is False
    with pytest.raises(OllamaRuntimeError):
        client.chat(messages=[{"role": "user", "content": "ping"}])
