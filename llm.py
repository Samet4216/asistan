"""
LLM Interface Module for Local Deep Learning Lab Assistant.
Direct Ollama HTTP API wrapper supporting JSON schemas, keep-alive control,
and explicit VRAM unloading.
"""

import json
import requests
import time
import threading
from typing import List, Dict, Any, Optional
import config


class LLMClient:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.MODEL
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.last_active_time = time.time()
        self.is_loaded = False
        self.auto_unload_enabled = True
        self.timeout_seconds = getattr(config, "IDLE_UNLOAD_TIMEOUT_SECONDS", 600)
        self._lock = threading.Lock()
        self._watchdog_thread = None
        self._start_watchdog()

    def _start_watchdog(self):
        """Starts a background daemon thread to monitor idle timeout and auto-unload VRAM."""
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="VRAMIdleWatchdog"
            )
            self._watchdog_thread.start()

    def _watchdog_loop(self):
        """Background loop that checks every 10 seconds if idle timeout is exceeded."""
        while True:
            time.sleep(10)
            if not self.auto_unload_enabled:
                continue
            with self._lock:
                if not self.is_loaded:
                    continue
                idle_sec = time.time() - self.last_active_time
                if idle_sec >= self.timeout_seconds:
                    self._unload_internal(reason="idle_timeout")

    def _touch_active(self):
        """Updates activity timestamp and marks model as loaded in VRAM."""
        with self._lock:
            self.last_active_time = time.time()
            self.is_loaded = True

    def set_model(self, model_name: str) -> None:
        """Switches the active model, unloading previous model if different to protect 6GB VRAM."""
        if not model_name or model_name == self.model:
            return
        with self._lock:
            if self.is_loaded:
                self.unload_model(self.model)
            self.model = model_name

    def unload_model(self, model_name: str) -> str:
        """Unloads a specific model from VRAM immediately."""
        try:
            requests.post(f"{self.base_url}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=5)
            self.is_loaded = False
            return f"Model {model_name} unloaded."
        except Exception as e:
            return f"Failed to unload {model_name}: {e}"

    def list_models(self) -> List[str]:
        """Queries local Ollama instance for installed models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 800,
        temperature: float = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Calls Ollama /api/chat with strict options.
        If `schema` is provided, Ollama enforces structured JSON output.
        """
        target_model = model or self.model
        if target_model != self.model:
            self.set_model(target_model)

        self._touch_active()
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
            self._touch_active()
            return data.get("message", {}).get("content", "").strip()

        except requests.exceptions.RequestException as e:
            return f"[LLM ERROR]: Failed to communicate with Ollama at {self.chat_endpoint}: {e}"

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = None,
        model: Optional[str] = None
    ):
        """Streams generation token by token from Ollama /api/chat."""
        target_model = model or self.model
        if target_model != self.model:
            self.set_model(target_model)

        self._touch_active()
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
            self._touch_active()
        except Exception as e:
            yield f"\n[LLM STREAM ERROR]: {e}"

    def _unload_internal(self, reason: str = "manual") -> str:
        payload = {
            "model": self.model,
            "keep_alive": 0
        }
        try:
            requests.post(f"{self.base_url}/api/generate", json=payload, timeout=5)
            self.is_loaded = False
            if reason == "idle_timeout":
                msg = f"[*] [Auto-Unload] {int(self.timeout_seconds / 60)} dakika boşta kalındığı için model ({self.model}) VRAM'den otomatik boşaltıldı (GPU soğutuldu)."
                print(msg)
                return msg
            else:
                msg = f"Model {self.model} unloaded from VRAM."
                print(f"[*] {msg}")
                return msg
        except Exception as e:
            err = f"Failed to unload model: {e}"
            print(f"[!] {err}")
            return err

    def unload(self) -> str:
        """Immediately unloads model weights from VRAM to reduce idle heat."""
        with self._lock:
            return self._unload_internal(reason="manual")

    def get_idle_status(self) -> Dict[str, Any]:
        """Returns the current idle timer and VRAM status."""
        with self._lock:
            idle_sec = int(time.time() - self.last_active_time)
            remaining_sec = max(0, self.timeout_seconds - idle_sec)
            return {
                "is_loaded": self.is_loaded,
                "idle_seconds": idle_sec,
                "timeout_seconds": self.timeout_seconds,
                "remaining_seconds": remaining_sec,
                "auto_unload_enabled": self.auto_unload_enabled
            }


# Global default client instance
client = LLMClient()


def generate(
    messages: List[Dict[str, str]],
    schema: Optional[Dict[str, Any]] = None,
    max_tokens: int = 800,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience functional interface."""
    return client.generate(messages=messages, schema=schema, max_tokens=max_tokens, temperature=temperature, model=model, **kwargs)


def stream_generate(
    messages: List[Dict[str, str]],
    max_tokens: int = 1500,
    temperature: float = None,
    model: Optional[str] = None
):
    """Convenience streaming functional interface."""
    return client.stream_generate(messages=messages, max_tokens=max_tokens, temperature=temperature, model=model)


def unload():
    """Convenience unload function."""
    client.unload()


def get_idle_status() -> Dict[str, Any]:
    """Convenience status function."""
    return client.get_idle_status()


def set_auto_unload(enabled: bool, timeout_seconds: int = None):
    """Configures automatic idle unload timer."""
    client.auto_unload_enabled = enabled
    if timeout_seconds:
        client.timeout_seconds = timeout_seconds


def is_model_available(model_name: str) -> bool:
    """Checks if a given model is already downloaded and ready in Ollama."""
    return client.is_model_available(model_name)


def list_models() -> List[str]:
    """Queries local Ollama instance for installed models."""
    return client.list_models()


def parse_deepseek_thought(raw_text: str) -> Dict[str, str]:
    """
    Separates DeepSeek-R1's `<think> ... </think>` internal reasoning block from the final response.
    Returns {"thought": "...", "response": "..."}.
    """
    import re
    if not raw_text:
        return {"thought": "", "response": ""}
    
    # Check complete <think> ... </think>
    match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        response = raw_text[match.end():].strip()
        return {"thought": thought, "response": response}
    
    # Check open <think> (still thinking or unclosed)
    if "<think>" in raw_text:
        parts = raw_text.split("<think>", 1)
        return {"thought": parts[1].strip(), "response": parts[0].strip()}

    return {"thought": "", "response": raw_text.strip()}


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
