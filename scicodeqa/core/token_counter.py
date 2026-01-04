import logging
import os
import time
from collections import deque
from logging.config import fileConfig
from typing import Optional

import requests
from joblib import Memory
from litellm import token_counter
from transformers import AutoTokenizer

fileConfig("logging.ini")
logger = logging.getLogger(__name__)

memory = Memory(location=".cache/token_counter", compress=False, verbose=0)


class TokenCounter:
    _gemini_requests = deque()  # Track request timestamps for rate limiting
    _gemini_rpm_limit = 200
    _anthropic_requests = deque()
    _anthropic_rpm_limit = 100

    def __init__(self, model: str):
        self.method, *model_parts = model.split("/")
        self.model = "/".join(model_parts)
        self.hf_tokenizer: Optional[object] = None

        self.huggingface_token_counter = memory.cache(
            self.huggingface_token_counter, ignore=["self"]
        )
        self._gemini_api_request = memory.cache(
            self._gemini_api_request, ignore=["self"]
        )
        self._anthropic_api_request = memory.cache(
            self._anthropic_api_request, ignore=["self"]
        )
        self._vllm_api_request = memory.cache(self._vllm_api_request, ignore=["self"])
        self.default_token_counter = memory.cache(
            self.default_token_counter, ignore=["self"]
        )

        if self.method == "hf":
            logger.info(f"Using HuggingFace tokenizer for {self.model}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
        elif self.method == "gemini":
            logger.info(f"Using Gemini tokenizer for {self.model}")
        elif self.method == "anthropic":
            logger.info(f"Using Anthropic tokenizer for {self.model}")
        elif self.method == "vllm":
            logger.info(f"Using VLLM tokenizer for {self.model}")
        else:
            logger.info(f"Using litellm token counter for {self.model}")

    def __call__(self, text: str) -> int:
        if len(text) == 0:
            return 0
        if self.method == "gemini":
            return self.gemini_token_counter(text, self.model)
        elif self.method == "anthropic":
            return self.anthropic_token_counter(text, self.model)
        elif self.method == "vllm":
            return self.vllm_token_counter(text, self.model)
        elif self.method == "hf":
            return self.huggingface_token_counter(text, self.model)
        else:
            return self.default_token_counter(text, self.model)

    def default_token_counter(self, text: str, model: str) -> int:
        return token_counter(text=text, model=model)

    def huggingface_token_counter(self, text: str, model: str) -> int:
        tokens = self.hf_tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)

    @classmethod
    def _rate_limit_gemini(cls):
        """Enforce 3000 RPM rate limit for Gemini API."""
        now = time.time()
        # Remove requests older than 60 seconds
        while cls._gemini_requests and cls._gemini_requests[0] < now - 60:
            cls._gemini_requests.popleft()
        # If at limit, wait until oldest request expires
        if len(cls._gemini_requests) >= cls._gemini_rpm_limit:
            sleep_time = cls._gemini_requests[0] + 60 - now
            if sleep_time > 0:
                logger.warning(
                    "Gemini API rate limit exceeded. "
                    f"Sleeping for {sleep_time:.2f} seconds."
                )
                time.sleep(sleep_time)
            cls._gemini_requests.popleft()
        cls._gemini_requests.append(time.time())

    def _gemini_api_request(self, text: str, model: str) -> int:
        """Make the actual API request to Gemini. This method is cached."""
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:countTokens"
        )
        headers = {"Content-Type": "application/json"}
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY is not set")
        params = {"key": gemini_api_key}
        payload = {"contents": [{"parts": [{"text": text}]}]}

        response = requests.post(url, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
        tokens = result.get("totalTokens", -1)
        if tokens == -1:
            raise ValueError(f"Failed to count tokens for {model}: {result}")
        return tokens

    def gemini_token_counter(self, text: str, model: str) -> int:
        """Count tokens with exponential backoff retry logic."""
        max_retries = 8
        base_delay = 1.0
        max_delay = 60.0

        for attempt in range(max_retries):
            try:
                self._rate_limit_gemini()
                return self._gemini_api_request(text, model)
            except requests.exceptions.RequestException as e:
                # Calculate current RPM
                now = time.time()
                recent_requests = sum(
                    1 for ts in self._gemini_requests if ts > now - 60
                )

                if attempt < max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Gemini API error (attempt {attempt + 1}/{max_retries}). "
                        "Current RPM: "
                        f"{recent_requests}/{TokenCounter._gemini_rpm_limit}. "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Gemini API error after {max_retries} attempts. "
                        "Current RPM: "
                        f"{recent_requests}/{TokenCounter._gemini_rpm_limit}."
                    )
                    raise e

    @classmethod
    def _rate_limit_anthropic(cls):
        """Enforce rate limit for Anthropic API."""
        now = time.time()
        # Remove requests older than 60 seconds
        while cls._anthropic_requests and cls._anthropic_requests[0] < now - 60:
            cls._anthropic_requests.popleft()
        # If at limit, wait until oldest request expires
        if len(cls._anthropic_requests) >= cls._anthropic_rpm_limit:
            sleep_time = cls._anthropic_requests[0] + 60 - now
            if sleep_time > 0:
                logger.warning(
                    "Anthropic API rate limit exceeded. "
                    f"Sleeping for {sleep_time:.2f} seconds."
                )
                time.sleep(sleep_time)
            cls._anthropic_requests.popleft()
        cls._anthropic_requests.append(time.time())

    def _anthropic_api_request(self, text: str, model: str) -> int:
        """Make the actual API request to Anthropic. This method is cached."""
        url = "https://api.anthropic.com/v1/messages/count_tokens"
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {"model": model, "messages": [{"role": "user", "content": text}]}

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        tokens = result.get("input_tokens", -1)
        if tokens == -1:
            raise ValueError(f"Failed to count tokens for {model}: {result}")
        return tokens

    def anthropic_token_counter(self, text: str, model: str) -> int:
        """Count tokens with exponential backoff retry logic."""
        max_retries = 8
        base_delay = 1.0
        max_delay = 60.0

        for attempt in range(max_retries):
            try:
                self._rate_limit_anthropic()
                return self._anthropic_api_request(text, model)
            except requests.exceptions.RequestException as e:
                # Calculate current RPM
                now = time.time()
                recent_requests = sum(
                    1 for ts in self._anthropic_requests if ts > now - 60
                )

                if attempt < max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Anthropic API error (attempt {attempt + 1}/{max_retries}). "
                        "Current RPM: "
                        f"{recent_requests}/{TokenCounter._anthropic_rpm_limit}. "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Anthropic API error after {max_retries} attempts. "
                        "Current RPM: "
                        f"{recent_requests}/{TokenCounter._anthropic_rpm_limit}."
                    )
                    raise e

    def _vllm_api_request(self, text: str, model: str) -> int:
        """Make the actual API request to VLLM. This method is cached."""
        vllm_api_base = os.environ.get("VLLM_API_BASE")
        if vllm_api_base is None:
            raise ValueError("VLLM_API_BASE is not set")

        # Remove /v1 suffix if present to construct the base URL
        base_url = vllm_api_base.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        url = f"{base_url}/tokenize"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "prompt": text,
            "add_special_tokens": True,
            "return_token_strs": False,
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        tokens = result.get("count", -1)
        if tokens == -1:
            raise ValueError(f"Failed to count tokens for {model}: {result}")
        return tokens

    def vllm_token_counter(self, text: str, model: str) -> int:
        """Count tokens using VLLM tokenize endpoint (no rate limiting needed)."""
        return self._vllm_api_request(text, model)
