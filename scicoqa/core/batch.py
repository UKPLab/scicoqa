import argparse
import json
import logging
import os
import time
from datetime import datetime
from logging.config import fileConfig
from pathlib import Path

from google import genai
from openai import OpenAI

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


class Batch:
    def __init__(self):
        self.google_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def list_batch_jobs(self, provider: str = "google"):
        if provider == "google":
            batch_jobs = self.google_client.batches.list()

            for b in batch_jobs.page:
                print(f"Job Name: {b.name}")
                print(f"  - Display Name: {b.display_name}")
                print(f"  - State: {b.state.name}")
                print(f"  - Create Time: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")
                if b.dest is not None:
                    if not b.dest.file_name:
                        full_job = self.google_client.batches.get(name=b.name)
                        if full_job.inlined_responses:
                            print(
                                "  - Type: Inline ({} responses)".format(
                                    len(full_job.inlined_responses)
                                )
                            )
                    else:
                        print(f"  - Type: File-based (Output: {b.dest.file_name})")
        elif provider == "openai":
            page = self.openai_client.batches.list()
            for b in page:
                print(f"ID: {b.id}")
                print(f"  - Status: {b.status}")
                created_date = datetime.fromtimestamp(b.created_at)
                print(f"  - Created: {created_date.strftime('%Y-%m-%d %H:%M:%S')}")
                for k, v in b.metadata.items():
                    print(f"  - {k}: {v}")
                if b.output_file_id:
                    print(f"  - Output File ID: {b.output_file_id}")
                if b.error_file_id:
                    print(f"  - Error File ID: {b.error_file_id}")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def cancel_batch_job(self, name: str, provider: str = "google"):
        if provider == "google":
            self.google_client.batches.cancel(name=name)
        elif provider == "openai":
            self.openai_client.batches.cancel(batch_id=name)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def poll(self, exp_dir: Path, wait_between_polls: int = 30):
        with open(exp_dir / "batch_job.json", "r") as f:
            batch_job = json.load(f)
        with open(exp_dir / "args.json", "r") as f:
            args = json.load(f)

        if args["model"].startswith("gemini"):
            self._poll_google(batch_job, exp_dir, wait_between_polls)
        elif args["model"].startswith("gpt"):
            self._poll_openai(batch_job, exp_dir, wait_between_polls)
        else:
            raise ValueError(f"Model {batch_job['model']} not supported")

    def _poll_google(
        self, batch_job: dict, exp_dir: Path, wait_between_polls: int = 30
    ):
        while True:
            batch = self.google_client.batches.get(name=batch_job["name"])
            if batch.state.name in ("JOB_STATE_SUCCEEDED",):
                break
            if batch.state.name in (
                "JOB_STATE_EXPIRED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
            ):
                raise ValueError(f"Batch job failed with state: {batch.state.name}")
            logger.info(
                f"Job not finished. Current state: {batch.state.name}. "
                f"Waiting {wait_between_polls} seconds..."
            )
            time.sleep(wait_between_polls)

        logger.info(f"Batch job finished with state: {batch.state.name}")

        # update the batch job file
        with open(exp_dir / "batch_job.json", "w") as f:
            f.write(batch.model_dump_json(indent=2))

        # create generations.jsonl file
        responses = []
        for R in batch.dest.inlined_responses:
            response = {}

            if len(R.response.candidates[0].content.parts) == 1:
                generation = R.response.candidates[0].content.parts[0].text
            elif len(R.response.candidates[0].content.parts) == 2:
                for part in R.response.candidates[0].content.parts:
                    if part.get("thought", False):
                        thinking = part.text
                    else:
                        generation = part.text
            else:
                raise ValueError(
                    "Unexpected number of parts: "
                    f"{len(R.response.candidates[0].content.parts)}"
                )

            response["id"] = R.response.response_id
            response["created"] = int(
                self.google_client.batches.get(name=batch.name).create_time.timestamp()
            )
            response["model"] = R.response.model_version
            response["object"] = None
            response["system_fingerprint"] = None
            response["choices"] = [
                {
                    "finish_reason": str(R.response.candidates[0].finish_reason),
                    "index": R.response.candidates[0].index,
                    "message": {
                        "content": generation,
                        "role": "assistant",
                        "thinking_blocks": [{"type": "thinking", "thinking": thinking}],
                    },
                }
            ]
            response["usage"] = {
                "completion_tokens": R.response.usage_metadata.candidates_token_count,
                "prompt_tokens": R.response.usage_metadata.prompt_token_count,
                "total_tokens": R.response.usage_metadata.total_token_count,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": None,
                    "audio_tokens": None,
                    "reasoning_tokens": R.response.usage_metadata.thoughts_token_count,
                    "rejected_prediction_tokens": None,
                    "text_tokens": None,
                },
            }
            response["vertex_ai_grounding_metadata"] = (
                R.response.vertex_ai_grounding_metadata
            )

            responses.append(response)

        with (
            open(exp_dir / "generations.jsonl", "w") as f_generations,
            open(exp_dir / "contexts.jsonl", "r") as f_contexts,
        ):
            for context, response in zip(f_contexts.readlines(), responses):
                generation = {**json.loads(context), "response": response}
                f_generations.write(json.dumps(generation) + "\n")

    def _poll_openai(
        self, batch_job: dict, exp_dir: Path, wait_between_polls: int = 30
    ):
        while True:
            batch = self.openai_client.batches.retrieve(batch_id=batch_job["id"])
            if batch.status in ("completed",):
                break
            if batch.status in ("failed", "expired", "cancelled"):
                raise ValueError(f"Batch job failed with status: {batch.status}")
            logger.info(
                f"Job not finished. Current status: {batch.status}. "
                f"Waiting {wait_between_polls} seconds..."
            )
            time.sleep(wait_between_polls)

        logger.info(f"Batch job finished with status: {batch.status}")

        # update the batch job file
        batch_dict = batch.model_dump()
        with open(exp_dir / "batch_job_completed.json", "w") as f:
            f.write(json.dumps(batch_dict, indent=2))

        # Download output JSONL and build generations
        if not batch.output_file_id:
            logger.warning("Completed OpenAI batch has no output_file_id.")
            if not batch.error_file_id:
                raise ValueError("Completed OpenAI batch has no error_file_id")
            logger.warning(
                "Completed OpenAI batch has no output_file_id. "
                f"Using error_file_id: {batch.error_file_id}"
            )
            content = self.openai_client.files.content(batch.error_file_id)
            data_bytes = content.read()
            logger.warning(f"Error Output: {data_bytes}")
            raise RuntimeError("Completed OpenAI resulted in Error.")

        content = self.openai_client.files.content(batch.output_file_id)
        data_bytes = content.read()
        responses = []
        for line in data_bytes.splitlines():
            if not line:
                continue
            entry = json.loads(line)
            body = entry.get("response", {}).get("body")
            if body is None:
                continue
            responses.append(body)

        with (
            open(exp_dir / "generations.jsonl", "w") as f_generations,
            open(exp_dir / "contexts.jsonl", "r") as f_contexts,
        ):
            for context, response in zip(f_contexts.readlines(), responses):
                generation = {**json.loads(context), "response": response}
                f_generations.write(json.dumps(generation) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Google Batch jobs for scicoqa"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # poll subcommand
    poll_parser = subparsers.add_parser(
        "poll", help="Poll a batch job until completion and materialize outputs"
    )
    poll_parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment directory containing batch_job.json and args.json",
    )
    poll_parser.add_argument(
        "--interval", type=int, default=30, help="Seconds to wait between polls"
    )

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List batch jobs")
    list_parser.add_argument(
        "--provider",
        choices=["google", "openai"],
        default="google",
        help="Provider backend for batch jobs",
    )

    # cancel subcommand
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a batch job")
    cancel_parser.add_argument(
        "target",
        type=str,
        help="Batch job name or experiment directory containing batch_job.json",
    )
    cancel_parser.add_argument(
        "--provider",
        choices=["google", "openai"],
        help=(
            "Provider backend. If omitted and --exp-dir is given, "
            "inferred from model name."
        ),
    )

    args = parser.parse_args()

    if args.command == "poll":
        batch = Batch()
        batch.poll(args.exp_dir, wait_between_polls=args.interval)
    elif args.command == "list":
        batch = Batch()
        batch.list_batch_jobs(provider=args.provider)
    elif args.command == "cancel":
        target_path = Path(args.target)
        if target_path.is_dir():
            batch = Batch()
            batch_job_path = target_path / "batch_job.json"
            if not batch_job_path.exists():
                raise ValueError(
                    "Could not load batch_job.json from provided directory"
                )
            with open(batch_job_path, "r") as f:
                batch_job = json.load(f)
            model_name = batch_job.get("model", "")
            provider = args.provider
            if provider is None:
                if model_name.startswith("models/gemini"):
                    provider = "google"
                else:
                    provider = "openai"
            batch.cancel_batch_job(batch_job["name"], provider=provider)
        else:
            provider = args.provider or "google"
            batch = Batch()
            batch.cancel_batch_job(args.target, provider=provider)


if __name__ == "__main__":
    main()
