import json
import logging
import string
from logging.config import fileConfig
from pathlib import Path

import yaml

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


class Prompt:
    def __init__(
        self, template: str, config_file: Path = Path("config") / "prompts.yaml"
    ):
        self.template = template
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

        if self.template not in self.config["prompts"]:
            raise ValueError(f"Template '{template}' not found in {config_file}")

        self.prompt = string.Template(self.config["prompts"][self.template])

        # Extract variables from the template
        self.prompt_vars = list(self.prompt.get_identifiers())

    def __call__(self, **kwargs) -> str:
        # remove any '<|endoftext|>' from the kwargs with endoftext
        for k, v in kwargs.items():
            if isinstance(v, str) and "<|endoftext|>" in v:
                kwargs[k] = v.replace("<|endoftext|>", "endoftext")
        return self.prompt.safe_substitute(**kwargs)

    def save(self, output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(
                {
                    "template": self.template,
                    "prompt": self.prompt.template,
                },
                f,
                indent=2,
            )
