"""Integration test for CLI and Ollama runtime.
"""
from unittest.mock import MagicMock, patch
import pytest

from personal_ai.runtime.ollama_client import OllamaClient
from personal_ai.app.cli import SYSTEM_ENGINEERING_PERSONA


def test_system_engineering_persona_rules():
    """Verify system persona contains strict negative constraints against low-quality output."""
    assert "yol haritası" in SYSTEM_ENGINEERING_PERSONA.lower()
    assert "bölüm 1/n" in SYSTEM_ENGINEERING_PERSONA.lower()
    assert "ascii" in SYSTEM_ENGINEERING_PERSONA.lower()
    assert "latex" in SYSTEM_ENGINEERING_PERSONA.lower()


def test_ollama_client_streaming_mocked():
    """Verify chat_stream correctly handles line-by-line JSON stream from Ollama."""
    client = OllamaClient(base_url="http://127.0.0.1:11434")

    mock_chunks = [
        b'{"message": {"content": "Merhaba "}, "done": false}\n',
        b'{"message": {"content": "dunya!"}, "done": false}\n',
        b'{"message": {"content": ""}, "done": true}\n',
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.__enter__.return_value = mock_response
    mock_response.iter_lines.return_value = mock_chunks

    with patch("requests.post", return_value=mock_response):
        tokens = list(client.chat_stream([{"role": "user", "content": "test"}]))
        assert tokens == ["Merhaba ", "dunya!"]


def test_ollama_client_non_streaming_mocked():
    """Verify non-streaming chat aggregates tokens properly."""
    client = OllamaClient(base_url="http://127.0.0.1:11434")

    mock_chunks = [
        b'{"message": {"content": "Deep learning mathematically optimizes parameters."}, "done": true}\n',
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.__enter__.return_value = mock_response
    mock_response.iter_lines.return_value = mock_chunks

    with patch("requests.post", return_value=mock_response):
        reply = client.chat([{"role": "user", "content": "what is dl?"}])
        assert "mathematically optimizes" in reply
