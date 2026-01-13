import datetime
import json
import logging
from logging.config import fileConfig

from scicoqa.core.args import BaseArgs
from scicoqa.core.data_iterator import (
    DiscrepancyIterator,
    GitHubIssueIterator,
    ReproducibilityReportDiscrepancyVerificationIterator,
    ReproducibilityReportIterator,
)
from scicoqa.core.llm import LLM
from scicoqa.core.prompt import Prompt

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


class BaseExperiment:
    def __init__(self, args: BaseArgs):
        self.args = args
        self.output_dir = args.output_dir
        self.generations_file = args.output_dir / "generations.jsonl"
        self.total_cost = 0

        # Initialize shared LLM and Prompt once
        self.llm = LLM(
            model=args.model,
            decoding_settings=args.decoding_config,
            output_file=args.output_dir / "llm.json",
        )
        self.prompt = Prompt(args.prompt)

        self.contexts = []

    def get_tokenizer(self) -> str:
        """Get tokenizer string from model config, fallback to model name."""
        return self.llm.model_config.get("tokenizer", self.llm.model_config["name"])

    def save(self, generations, metadata, prompt):
        if self.args.debug:
            logger.debug("Skipping saving because in debug mode")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generations = self._convert_datetime_to_string_nested_dict(generations)
        with open(self.generations_file, "a") as f:
            f.write(json.dumps(generations) + "\n")

        metadata = self._convert_datetime_to_string_nested_dict(metadata)
        with open(self.generations_file.parent / "metadata.jsonl", "a") as f:
            f.write(json.dumps(metadata) + "\n")

        self.args.save(self.output_dir / "args.json")
        self.llm.save()
        self.prompt.save(self.output_dir / "prompt.json")
        with open(self.output_dir / "prompts.jsonl", "a") as f:
            f.write(json.dumps({"prompt": prompt}) + "\n")

    def _convert_datetime_to_string_nested_dict(self, d):
        if isinstance(d, dict):
            return {
                k: self._convert_datetime_to_string_nested_dict(v) for k, v in d.items()
            }
        elif isinstance(d, (datetime.datetime, datetime.date)):
            return d.isoformat()
        return d

    def print_response(self, response):
        raw_response = response["choices"][0]["message"]["content"]
        logger.info(f"Response: {raw_response}")

    def _iter_items(self):
        """
        Yield dicts with keys:
          - prompt: str
          - pbar: tqdm instance
          - context: dict to merge into generations
          - log_label: str to include in logs
        """
        raise NotImplementedError

    def _format_generations(self, context: dict, response: dict) -> dict:
        # Default: merge context with response
        return {**context, "response": response}

    def _log_after_item(self, log_label: str):
        logger.info(f"Processed {log_label}, cost: {self.total_cost}")

    def __call__(self):
        for item in self._iter_items():
            prompt = item["prompt"]
            pbar = item["pbar"]
            context = item["context"]
            log_label = item["log_label"]
            try:
                response, metadata = self.llm(prompt, **self.args.llm_kwargs)
            except Exception as e:
                logger.exception(f"Error processing {log_label}")
                response = "ERROR"
                if self.args.raise_on_error:
                    raise e
                continue

            # In batch mode, calls return (None, None). Defer saving and cost.
            if response is None:
                self.contexts.append(context)
                continue

            generations = self._format_generations(context, response)
            self.total_cost += metadata["cost"]
            try:
                pbar.set_postfix({"cost": self.total_cost})
            except Exception:
                pass

            self.save(generations, metadata, prompt)
            self._log_after_item(log_label)

            if self.args.debug:
                self.print_response(response)

class ReproducibilityReportExperiment(BaseExperiment):
    def __init__(self, args: BaseArgs):
        super().__init__(args)
        self.reproducibility_report_iterator = ReproducibilityReportIterator(
            data_config_path=args.data_config_file,
            output_dir=args.output_dir,
            prompt=self.prompt,
            model=args.model,
            tokenizer=self.get_tokenizer(),
            model_max_tokens=self.llm.model_config["max_context_size"],
            debug=args.debug,
            debug_break_after=args.debug_break_after,
            iterate_over=args.iterate_over,
        )

    def _iter_items(self):
        for paper, prompt, pbar in self.reproducibility_report_iterator(
            **self.args.iterator_kwargs
        ):
            context = {"paper": paper}
            yield {
                "prompt": prompt,
                "pbar": pbar,
                "context": context,
                "log_label": paper["id"],
            }


class SyntheticDiscrepancyExperiment(BaseExperiment):
    def __init__(self, args: BaseArgs):
        super().__init__(args)
        from scicoqa.core.data_iterator import SyntheticDiscrepancyIterator

        self.synthetic_discrepancy_iterator = SyntheticDiscrepancyIterator(
            data_config_path=args.data_config_file,
            output_dir=args.output_dir,
            prompt=self.prompt,
            model=args.model,
            tokenizer=self.get_tokenizer(),
            model_max_tokens=self.llm.model_config["max_context_size"],
            debug=args.debug,
            debug_break_after=args.debug_break_after,
            data_config_section=getattr(
                args, "data_config_section", "synthetic_discrepancies"
            ),
        )

    def _iter_items(self):
        for paper, prompt, pbar in self.synthetic_discrepancy_iterator(
            **self.args.iterator_kwargs
        ):
            context = {"paper": paper}
            yield {
                "prompt": prompt,
                "pbar": pbar,
                "context": context,
                "log_label": paper["id"],
            }

class GitHubIssueExperiment(BaseExperiment):
    def __init__(self, args: BaseArgs):
        super().__init__(args)
        self.issue_iterator = GitHubIssueIterator(
            db_path=args.db_path,
            output_dir=args.output_dir,
            prompt=self.prompt,
            model=args.model,
            tokenizer=self.get_tokenizer(),
            model_max_tokens=self.llm.model_config["max_context_size"],
            debug=args.debug,
            debug_break_after=args.debug_break_after,
        )

    def _iter_items(self):
        for issue, prompt, pbar in self.issue_iterator(**self.args.iterator_kwargs):
            # Keep a compact issue payload (id/title/body)
            compact_issue = {
                "id": issue["id"],
                "title": issue["title"],
                "body": issue["body"],
            }
            context = {"issue": compact_issue}
            yield {
                "prompt": prompt,
                "pbar": pbar,
                "context": context,
                "log_label": f"issue {issue['id']}",
            }


class DiscrepancyExperiment(BaseExperiment):
    def __init__(self, args: BaseArgs):
        super().__init__(args)
        self.discrepancy_iterator = DiscrepancyIterator(
            discrepancy_file=args.discrepancy_file,
            output_dir=args.output_dir,
            prompt=self.prompt,
            model=args.model,
            tokenizer=self.get_tokenizer(),
            model_max_tokens=self.llm.model_config["max_context_size"],
            db_path=args.db_path,
            debug=args.debug,
            debug_break_after=args.debug_break_after,
        )

    def _iter_items(self):
        for discrepancy_issue, prompt, pbar in self.discrepancy_iterator(
            **self.args.iterator_kwargs
        ):
            context = {"discrepancy_issue": discrepancy_issue}
            yield {
                "prompt": prompt,
                "pbar": pbar,
                "context": context,
                "log_label": f"discrepancy issue {discrepancy_issue['issue_id']}",
            }


class ReproducibilityReportDiscrepancyVerificationExperiment(BaseExperiment):
    def __init__(self, args: BaseArgs):
        super().__init__(args)
        self.verification_iterator = (
            ReproducibilityReportDiscrepancyVerificationIterator(
                predictions_and_classifications_file=(
                    args.predictions_and_classifications_file
                ),
                data_config_path=args.data_config_file,
                output_dir=args.output_dir,
                prompt=self.prompt,
                model=args.model,
                tokenizer=self.get_tokenizer(),
                model_max_tokens=self.llm.model_config["max_context_size"],
                debug=args.debug,
                debug_break_after=args.debug_break_after,
            )
        )

    def _iter_items(self):
        for verification_item, prompt, pbar in self.verification_iterator(
            **self.args.iterator_kwargs
        ):
            context = {"verification_item": verification_item}
            yield {
                "prompt": prompt,
                "pbar": pbar,
                "context": context,
                "log_label": f"verification {verification_item['classification_key']}",
            }
