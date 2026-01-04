import inspect
import json
import logging
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path
from typing import Callable

from git import Repo

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BaseArgs:
    debug: bool = False
    debug_break_after: int = 1
    raise_on_error: bool = True

    def __post_init__(self):
        self.repo = Repo(search_parent_directories=True)
        if not self.debug and self.repo.is_dirty():
            raise RuntimeError("🧹 Repo is dirty, commit changes before running")

    @property
    def git_commit_hash(self):
        return self.repo.head.object.hexsha

    def save(self, output_file: Path):
        """Save args to JSON file in output directory."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving args to {output_file}")

        args_dict = {}
        for name, member in inspect.getmembers(self):
            if not name.startswith("__") and not isinstance(member, Callable):
                try:
                    json.dumps(member)
                    args_dict[name] = member
                except (TypeError, ValueError):
                    args_dict[name] = str(member)

        with open(output_file, "w") as f:
            json.dump(args_dict, f, indent=2)
        logger.info(f"Saved args to {output_file}")


@dataclass(kw_only=True)
class InferenceArgs(BaseArgs):
    model: str
    prompt: str
    dir_prefix: str
    decoding_config: str = "high_temperature"
    override: bool = False
    exp_id: str | None = None
    dir_suffix: str | None = None

    def __post_init__(self):
        super().__post_init__()

    def generate_next_id(self, parent_dir: Path, prefix: str) -> str:
        """Generate the next sequential ID with given prefix e.g., qgen-001"""

        if not parent_dir.exists():
            logger.debug(
                f"Parent directory {parent_dir} does not exist, using {prefix}-001"
            )
            return f"{prefix}-001"

        # Find existing directories with the given prefix
        existing_dirs = [
            d
            for d in parent_dir.iterdir()
            if d.is_dir() and d.name.startswith(f"{prefix}-")
        ]

        # Extract numbers from existing directories
        existing_numbers = []
        for dir_path in existing_dirs:
            try:
                number = int(dir_path.name.split("-")[1])
                existing_numbers.append(number)
            except (IndexError, ValueError):
                continue

        # Get next number
        next_number = max(existing_numbers, default=0) + 1
        return f"{prefix}-{next_number:03d}"

    def get_exp_dir_name(self, parent_dir: Path) -> str:
        """Get experiment directory name, generating it if exp_id is None"""
        if self.exp_id is None:
            logger.debug("exp_id is None, generating next ID")
            if not self.dir_prefix:
                raise ValueError(
                    "dir_prefix must be set in child class to auto-generate exp_id"
                )
            self.exp_id = self.generate_next_id(parent_dir, self.dir_prefix)
            # Auto-set dir_suffix to model name if not explicitly provided
            if hasattr(self, "dir_suffix"):
                if self.dir_suffix:
                    self.exp_id += f"-{self.dir_suffix}"
                elif hasattr(self, "model"):
                    # Use model name as suffix if dir_suffix is None/empty
                    self.exp_id += f"-{self.model}"
        logger.debug(f"Using exp_id: {self.exp_id}")
        return self.exp_id
