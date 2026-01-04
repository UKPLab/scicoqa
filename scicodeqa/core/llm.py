import asyncio
import hashlib
import json
import logging
import os
import time
from logging.config import fileConfig
from pathlib import Path

import yaml
from joblib import Memory
from litellm import (
    APIConnectionError,
    Timeout,
    acompletion,
    completion,
    completion_cost,
)
from tqdm.asyncio import tqdm_asyncio

fileConfig("logging.ini")
logger = logging.getLogger(__name__)

memory = Memory(location=".cache/llm", compress=False, verbose=0)


class LLM:
    def __init__(
        self,
        model: str,
        decoding_settings: dict,
        output_file: Path,
        config_file: Path = Path("config") / "models.yaml",
    ):
        self.model = model
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

        if model not in self.config["models"]:
            raise ValueError(
                f"Model {model} not found in {config_file}."
                f"Available models: {list(self.config['models'].keys())}"
            )
        self.model_config = self.config["models"][model]
        logger.debug(f"Model config: {self.model_config}")
        rate_limits = self.model_config.get("rate_limits", {})
        self.tokens_per_minute = rate_limits.get("tokens_per_minute", 0)
        self.requests_per_minute = rate_limits.get("requests_per_minute", 0)

        self.inference_config = self.config["inference_config"]
        logger.debug(f"Inference config: {self.inference_config}")
        self.decoding_config = self.config["decoding_config"][decoding_settings]
        logger.debug(f"Decoding config: {self.decoding_config}")

        self.output_file = output_file
        if self.output_file.exists():
            with open(self.output_file, "r") as f:
                previous_save = json.load(f)

            assert LLM._deep_equal(previous_save["model_config"], self.model_config), (
                "Model configuration has changed"
            )
            assert LLM._deep_equal(
                previous_save["decoding_config"], self.decoding_config
            ), "Decoding configuration has changed"
            assert LLM._deep_equal(
                previous_save["inference_config"], self.inference_config
            ), "Inference configuration has changed"
            self.calls = previous_save["calls"]
        else:
            self.calls = []

    def _get_rates(self) -> tuple[int, int]:
        """Count the number of requests and tokens in the last minute."""
        now = time.time()
        requests = 0
        tokens = 0
        for call in self.calls:
            if call["created"] > now - 60:
                requests += 1
                tokens += call["usage"]["total_tokens"]

        return requests, tokens

    def check_rate_limit(self):
        if self.tokens_per_minute > 0 or self.requests_per_minute > 0:
            while True:
                requests_last_minute, tokens_last_minute = self._get_rates()

                if (
                    self.requests_per_minute > 0
                    and requests_last_minute
                    > self.requests_per_minute * self.config["rate_limit_buffer"]
                ) or (
                    self.tokens_per_minute > 0
                    and tokens_last_minute
                    > self.tokens_per_minute * self.config["rate_limit_buffer"]
                ):
                    logger.warning(
                        f"Close to rate limit for {self.model}. "
                        f"Requests last minute: {requests_last_minute}. "
                        f"Tokens last minute: {tokens_last_minute}. "
                        f"Cooling off for 6 seconds."
                    )
                    time.sleep(6)
                else:
                    break

    @staticmethod
    @memory.cache
    def _completion(
        model: str,
        prompt: str,
        **completion_kwargs,
    ) -> dict:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **completion_kwargs,
        )
        response = response.model_dump()
        return response

    def __call__(
        self,
        prompt: str,
    ) -> tuple[dict, dict]:
        self.check_rate_limit()

        tik = time.time()
        kwargs = {}
        if self.model_config["name"].startswith("ollama"):
            kwargs["base_url"] = os.environ.get(
                "OLLAMA_API_BASE",
                os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            )
            kwargs["num_ctx"] = self.model_config["max_context_size"]
            logger.debug(f"Using {kwargs['num_ctx']} tokens for {self.model}.")
        elif self.model_config["name"].startswith("hosted_vllm/"):
            kwargs["api_base"] = os.environ.get(
                "VLLM_API_BASE", "http://localhost:11434/v1"
            )
            logger.debug(f"Using {kwargs['api_base']} for {self.model}.")
        elif self.model_config["name"].startswith("openai/"):
            # set prompt_cache_key based on the beginning of the prompt and create a
            # cache key
            prompt_start = " ".join(prompt.split(" ")[:128])
            prompt_hash = hashlib.sha256(prompt_start.encode()).hexdigest()
            kwargs["prompt_cache_key"] = prompt_hash

        # Pass request_params from model config if it exists
        if "request_params" in self.model_config:
            if any(key in kwargs for key in self.model_config["request_params"]):
                raise ValueError(
                    f"Request params {self.model_config['request_params']} conflict "
                    f"with kwargs {kwargs} for model {self.model_config['name']}"
                )
            kwargs.update(self.model_config["request_params"])

        response_has_error = False
        try:
            response = self._completion(
                self.model_config["name"],
                prompt,
                **self.inference_config,
                **self.decoding_config,
                **kwargs,
            )
        except (APIConnectionError, Timeout) as e:
            logger.error(f"Error calling {self.model_config['name']}: {e}")
            if isinstance(e, Timeout):
                mock_response = "ERROR: Timeout"
            elif isinstance(e, APIConnectionError):
                mock_response = "ERROR: APIConnectionError"
            response = completion(
                model=self.model_config["name"],
                messages=[{"role": "user", "content": prompt}],
                mock_response=mock_response,
            )
            response = response.model_dump()
            response_has_error = True
        except Exception as e:
            raise e
        tok = time.time()

        self.calls.append(
            {
                "created": time.time(),
                "usage": response["usage"],
            }
        )

        if response_has_error:
            cost = 0
        else:
            try:
                cost = completion_cost(response)
            except Exception as e:
                logger.error(
                    f"Error calculating cost for {self.model_config['name']}: {e}"
                )
                cost = 0

        metadata = {"inference_time": tok - tik, "cost": cost}

        return response, metadata

    @staticmethod
    def _deep_equal(dict1, dict2):
        if not isinstance(dict1, dict) or not isinstance(dict2, dict):
            return dict1 == dict2
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        return all(LLM._deep_equal(dict1[k], dict2[k]) for k in dict1.keys())

    def save(self):
        with open(self.output_file, "w") as f:
            json.dump(
                {
                    "model_config": self.model_config,
                    "decoding_config": self.decoding_config,
                    "inference_config": self.inference_config,
                    "calls": self.calls,
                },
                f,
                indent=2,
            )


class AsyncVLLM:
    """Async LLM client for high-throughput vLLM inference."""

    def __init__(
        self,
        model: str,
        decoding_settings: str,
        output_file: Path,
        config_file: Path = Path("config") / "models.yaml",
        vllm_server_url: str | None = None,
        concurrency: int = 128,
        seed: int = 42,
    ):
        self.model = model
        self.output_file = output_file
        self.concurrency = concurrency
        self.seed = seed

        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

        if model not in self.config["models"]:
            raise ValueError(
                f"Model {model} not found in {config_file}. "
                f"Available models: {list(self.config['models'].keys())}"
            )
        self.model_config = self.config["models"][model]
        logger.debug(f"Model config: {self.model_config}")

        self.decoding_config = self.config["decoding_config"].get(decoding_settings, {})
        logger.debug(f"Decoding config: {self.decoding_config}")

        # Use provided vllm_server_url or get from env
        self.api_base = vllm_server_url or os.environ.get(
            "VLLM_API_BASE", "http://localhost:11434/v1"
        )
        logger.info(f"Using vLLM server at: {self.api_base}")

        self.calls = []

    async def _run_single_completion(
        self,
        prompt: str,
        idx: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, dict]:
        """Run a single async completion with semaphore for concurrency control."""
        async with semaphore:
            # Build kwargs from decoding config and model request_params
            kwargs = {}

            # Add decoding config params
            if "temperature" in self.decoding_config:
                kwargs["temperature"] = self.decoding_config["temperature"]
            if "top_p" in self.decoding_config:
                kwargs["top_p"] = self.decoding_config["top_p"]
            if "reasoning_effort" in self.decoding_config:
                kwargs["reasoning_effort"] = self.decoding_config["reasoning_effort"]

            # Add model-specific request params
            request_params = self.model_config.get("request_params", {})
            for key, value in request_params.items():
                if key not in kwargs:
                    kwargs[key] = value

            try:
                response = await acompletion(
                    api_base=self.api_base,
                    model=self.model_config["name"],
                    messages=[{"role": "user", "content": prompt}],
                    timeout=kwargs.pop("request_timeout", 3600),
                    seed=self.seed,
                    **kwargs,
                )
                result = response.model_dump()
            except Exception as e:
                logger.error(f"Error in completion {idx}: {e}")
                result = {
                    "error": str(e),
                    "choices": [{"message": {"content": f"ERROR: {e}"}}],
                }

            return idx, result

    async def batch_completion(
        self,
        prompts: list[str],
        show_progress: bool = True,
    ) -> list[dict]:
        """Run batch completions with high concurrency optimized for vLLM throughput."""
        semaphore = asyncio.Semaphore(self.concurrency)

        tasks = [
            self._run_single_completion(prompt, i, semaphore)
            for i, prompt in enumerate(prompts)
        ]

        # Gather with progress bar
        indexed_results = await tqdm_asyncio.gather(
            *tasks,
            desc="vLLM batch",
            total=len(prompts),
            disable=not show_progress,
        )

        # Sort by index to ensure correct order
        indexed_results.sort(key=lambda x: x[0])
        results = [r for _, r in indexed_results]

        # Track calls for saving
        self.calls.append(
            {
                "created": time.time(),
                "num_prompts": len(prompts),
                "concurrency": self.concurrency,
            }
        )

        return results

    def save(self):
        """Save configuration and call history."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(
                {
                    "model_config": self.model_config,
                    "decoding_config": self.decoding_config,
                    "api_base": self.api_base,
                    "concurrency": self.concurrency,
                    "seed": self.seed,
                    "calls": self.calls,
                },
                f,
                indent=2,
            )
