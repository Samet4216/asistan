"""
LLM Interface Module for Local Deep Learning Lab Assistant.
Direct Ollama HTTP API wrapper supporting JSON schemas, keep-alive control,
and explicit VRAM unloading.
"""

import json
import requests
from typing import List, Dict, Any, Optional
import config


class LLMClient:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.MODEL
        self.chat_endpoint = f"{self.base_url}/api/chat"

    def generate(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 800,
        temperature: float = None
    ) -> str:
        """
        Calls Ollama /api/chat with strict options.
        If `schema` is provided, Ollama enforces structured JSON output.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": config.KEEP_ALIVE,
            "options": {
                "num_ctx": config.NUM_CTX,
                "temperature": temperature if temperature is not None else config.TEMPERATURE,
                "num_predict": max_tokens
            }
        }

        if schema:
            payload["format"] = schema

        try:
            response = requests.post(
                self.chat_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()

        except requests.exceptions.RequestException as e:
            return f"[LLM ERROR]: Failed to communicate with Ollama at {self.chat_endpoint}: {e}"

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = None
    ):
        """Streams generation token by token from Ollama /api/chat."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": config.KEEP_ALIVE,
            "options": {
                "num_ctx": config.NUM_CTX,
                "temperature": temperature if temperature is not None else config.TEMPERATURE,
                "num_predict": max_tokens
            }
        }
        try:
            response = requests.post(
                self.chat_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except Exception as e:
            yield f"\n[LLM STREAM ERROR]: {e}"

    def unload(self):
        """Immediately unloads model weights from VRAM to reduce idle heat."""
        payload = {
            "model": self.model,
            "keep_alive": 0
        }
        try:
            requests.post(f"{self.base_url}/api/generate", json=payload, timeout=5)
            print(f"[*] Model {self.model} unloaded from VRAM.")
        except Exception as e:
            print(f"[!] Failed to unload model: {e}")


# Global default client instance
client = LLMClient()


def generate(messages: List[Dict[str, str]], schema: Optional[Dict[str, Any]] = None, max_tokens: int = 800) -> str:
    """Convenience functional interface."""
    return client.generate(messages=messages, schema=schema, max_tokens=max_tokens)


def stream_generate(messages: List[Dict[str, str]], max_tokens: int = 1500, temperature: float = None):
    """Convenience streaming functional interface."""
    return client.stream_generate(messages=messages, max_tokens=max_tokens, temperature=temperature)


def unload():
    """Convenience unload function."""
    client.unload()


if __name__ == "__main__":
    print("[*] Testing llm.py with simple greeting...")
    test_msgs = [
        {"role": "system", "content": "You are a concise academic deep learning assistant. Answer in 1 sentence."},
        {"role": "user", "content": "What is the primary benefit of residual connections in deep networks?"}
    ]
    resp = generate(test_msgs)
    print(f"\n[Response]:\n{resp}\n")

    print("[*] Testing JSON schema enforcement...")
    test_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "benefit": {"type": "string"}
        },
        "required": ["topic", "benefit"]
    }
    schema_msgs = [
        {"role": "user", "content": "Summarize batch normalization benefits in JSON."}
    ]
    json_resp = generate(schema_msgs, schema=test_schema)
    print(f"\n[JSON Response]:\n{json_resp}\n")
