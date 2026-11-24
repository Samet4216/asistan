"""Ollama Local Model Runtime Client.

Provides synchronous and streaming inference backends for local models
configured via `configs/model.yaml`.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import requests
import yaml


class OllamaRuntimeError(Exception):
    """Raised when communication with Ollama backend fails."""
    pass


class OllamaClient:
    """Encapsulates Ollama HTTP API interactions with configurable parameters."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self.config_path = Path(config_path or "configs/model.yaml")
        self.config = self._load_config()
        self.base_url = (base_url or self.config.get("ollama_base_url", "http://127.0.0.1:11434")).rstrip("/")
        self.default_model = default_model or self.config.get("default_model", "qwen2.5:7b-instruct-q4_K_M")
        self.default_temperature = float(self.config.get("temperature", 0.65))
        self.default_top_p = float(self.config.get("top_p", 0.90))
        self.default_top_k = int(self.config.get("top_k", 40))
        self.default_repeat_penalty = float(self.config.get("repeat_penalty", 1.10))
        self.default_num_ctx = int(self.config.get("num_ctx", 8192))

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from YAML file or returns sensible defaults."""
        if self.config_path.exists():
            try:
                data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
                return data.get("model", {}) if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def is_alive(self) -> bool:
        """Verifies if the Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Returns the list of locally available model tags."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m.get("name") for m in models if "name" in m]
            return []
        except Exception as e:
            raise OllamaRuntimeError(f"Failed to fetch models from {self.base_url}: {e}")

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streams assistant tokens from Ollama chat endpoint."""
        target_model = model or self.default_model
        temp = temperature if temperature is not None else self.default_temperature
        ctx = num_ctx if num_ctx is not None else self.default_num_ctx

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)

        payload = {
            "model": target_model,
            "messages": formatted_messages,
            "stream": True,
            "options": {
                "temperature": temp,
                "top_p": self.default_top_p,
                "top_k": self.default_top_k,
                "repeat_penalty": self.default_repeat_penalty,
                "num_ctx": ctx
            }
        }

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(10, 180)
            ) as resp:
                if resp.status_code != 200:
                    raise OllamaRuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        import json
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
        except requests.exceptions.ConnectionError:
            raise OllamaRuntimeError(
                f"Could not connect to Ollama at {self.base_url}. Please ensure Ollama is running."
            )
        except Exception as e:
            raise OllamaRuntimeError(f"Inference streaming error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Collects the full streamed response into a single string."""
        collected = []
        for token in self.chat_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt
        ):
            collected.append(token)
        return "".join(collected)
