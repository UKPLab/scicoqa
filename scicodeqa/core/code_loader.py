import logging
import os
import shutil
from datetime import datetime
from logging.config import fileConfig
from typing import Callable

import git
import nbconvert
import nbformat

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


class CodeLoader:
    def __init__(
        self,
        github_url: str,
        max_file_size_mb: float = 1.0,
        raw_repo_dir: str = "data/repos-raw",
        target_date: str | None = None,
    ):
        date_info = f" at date {target_date}" if target_date else ""
        logger.info(
            f"Initializing CodeLoader for {github_url} with max file size "
            f"{max_file_size_mb} MB and raw repo dir {raw_repo_dir}{date_info}"
        )
        self.github_url = github_url
        self.max_file_size_mb = max_file_size_mb
        self.raw_repo_dir = raw_repo_dir
        self.target_date = target_date
        self.repo_path = os.path.join(self.raw_repo_dir, self.github_url_to_repo_name)

        self.clone_repo()
        self.files = self._get_files()

    @property
    def github_url_to_repo_name(self):
        base_name = (
            self.github_url.rstrip("/").split("/")[-2]
            + "__"
            + self.github_url.rstrip("/").split("/")[-1]
        )
        if self.target_date:
            # Add date suffix to directory name to avoid conflicts
            date_suffix = (
                self.target_date.replace("-", "")
                .replace(":", "")
                .replace(" ", "_")
                .replace(".000Z", "")
                .replace("+0000", "")
            )
            base_name += f"__{date_suffix}"
        return base_name

    @property
    def repo_permalink(self):
        """Return a permalink to the repository at the current checked out commit."""
        try:
            repo = git.Repo(self.repo_path)
            current_commit = repo.head.commit.hexsha

            url_parts = self.github_url.rstrip("/").rstrip(".git").split("/")
            owner = url_parts[-2]
            repo_name = url_parts[-1]

            return f"https://github.com/{owner}/{repo_name}/tree/{current_commit}"
        except (ValueError, git.BadName, Exception) as e:
            logger.warning(f"Failed to get repository permalink: {e}")
            return self.github_url

    def clone_repo(self):
        if os.path.exists(self.repo_path):
            logger.info(f"Repository already exists at {self.repo_path}")

            # Always validate repository integrity first
            try:
                repo = git.Repo(self.repo_path)

                # Verify repository health by checking if HEAD is accessible
                try:
                    _ = repo.head.commit.hexsha
                except (ValueError, git.BadName) as e:
                    logger.warning(
                        "Repository has missing or corrupted commits at "
                        f"{self.repo_path}, removing and re-cloning. Error: {e}",
                    )
                    shutil.rmtree(self.repo_path)
                    self.clone_repo()  # Recursive call to re-clone
                    return

                # No need to checkout again - the target_date is encoded in the
                # directory path, so if this directory exists, it's already at the
                # correct commit
                logger.info(
                    "Repository already exists and checked out at correct state"
                )

            except (git.InvalidGitRepositoryError, git.GitCommandError) as e:
                logger.warning(
                    f"Invalid or corrupted git repository at {self.repo_path}, "
                    f"removing and re-cloning. Error: {e}"
                )
                shutil.rmtree(self.repo_path)
                self.clone_repo()  # Recursive call to re-clone
                return
            return

        logger.info(f"Cloning repo {self.github_url} to {self.repo_path}")
        os.makedirs(self.raw_repo_dir, exist_ok=True)
        repo = git.Repo.clone_from(
            self.github_url,
            self.repo_path,
        )

        # If target_date is specified, checkout the repository at that date
        if self.target_date:
            self._checkout_at_date(repo)

        # Clean up the repository
        self._cleanup_repo()

    def _cleanup_repo(self):
        """Remove docs/test directories, convert notebooks, and remove large files."""
        # remove any directories like docs, test, etc.
        for root, dirs, _ in os.walk(self.repo_path):
            # CRITICAL: Skip .git directory to avoid corrupting the repository
            if ".git" in dirs:
                dirs.remove(".git")

            # Create a copy of dirs to avoid modification during iteration
            dirs_to_remove = [
                dir
                for dir in dirs
                if dir in ["docs", "doc", "test", "tests", "example", "examples"]
            ]
            for dir in dirs_to_remove:
                dir_path = os.path.join(root, dir)
                logger.info(f"Removing directory: {dir_path}")
                shutil.rmtree(dir_path)
                # Remove from dirs to prevent os.walk from trying to recurse into
                # deleted dir
                dirs.remove(dir)

        # convert any jupyter notebooks to python files
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory
            if ".git" in dirs:
                dirs.remove(".git")

            for file in files:
                if file.endswith(".ipynb"):
                    logger.info(f"Converting Jupyter Notebook {file} to .py")
                    try:
                        nb = nbformat.read(os.path.join(root, file), as_version=4)
                        # Clear outputs
                        for cell in nb.cells:
                            if cell.get("cell_type") == "code":
                                cell["outputs"] = []
                                cell["execution_count"] = None

                        # Convert to .py
                        exporter = nbconvert.PythonExporter()
                        source, _ = exporter.from_notebook_node(nb)
                        # add a comment to the top of the file,
                        # stating that it was converted from a jupyter notebook
                        source = (
                            "# This file was converted from a jupyter notebook "
                            f"called {file}. All outputs have been removed.\n{source}"
                        )
                        with open(
                            os.path.join(root, file.replace(".ipynb", ".py")), "w"
                        ) as f:
                            f.write(source)
                        # remove the original notebook
                        os.remove(os.path.join(root, file))
                    except Exception as e:
                        logger.warning(f"Failed to convert notebook {file}: {e}")
                        raise e

        # remove any files larger than max_file_size_mb
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory
            if ".git" in dirs:
                dirs.remove(".git")

            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                except FileNotFoundError as e:
                    logger.warning(f"Failed to get size of {file_path}: {e}")
                    continue
                if file_size > self.mb_to_bytes(self.max_file_size_mb):
                    logger.info(f"Removing large file: {file_path}")
                    os.remove(file_path)

    def _checkout_at_date(self, repo: git.Repo):
        """Checkout the repository at the specified target date."""
        # Reset to clean state first
        # discard any local modifications from previous cleanup
        logger.info("Resetting repository to clean state")
        repo.git.reset("--hard")
        repo.git.clean("-fd")  # Remove untracked files and directories

        # Parse the target date
        target_datetime = datetime.fromisoformat(
            self.target_date.replace("Z", "+00:00")
        )
        logger.info(f"Checking out repository at date: {target_datetime}")

        # Use the current active branch - much simpler!
        try:
            current_branch = repo.active_branch.name
            logger.info(f"Using current active branch: {current_branch}")
        except TypeError:
            # In detached HEAD state, use HEAD
            current_branch = "HEAD"
            logger.info("Repository is in detached HEAD state, using HEAD")

        # Get commits from the current branch up to the target date
        commits = list(
            repo.iter_commits(rev=current_branch, until=target_datetime, max_count=1)
        )

        if not commits:
            logger.warning(
                f"No commits found before {target_datetime}, using latest commit"
            )
            commits = list(repo.iter_commits(rev=current_branch, max_count=1))

        if commits:
            target_commit = commits[0]
            commit_date = datetime.fromtimestamp(target_commit.committed_date)
            logger.info(
                f"Checking out commit {target_commit.hexsha[:8]} from {commit_date}"
            )
            try:
                repo.git.checkout(target_commit.hexsha)
            except (git.GitCommandError, ValueError) as e:
                logger.error(
                    f"Failed to checkout commit {target_commit.hexsha[:8]}: {e}. "
                    f"Repository may be corrupted or have missing objects."
                )
                raise
        else:
            logger.error("No commits found in repository")

    def _get_files(self):
        files = {}
        for root, _, _files in os.walk(self.repo_path):
            for file in _files:
                file_path = os.path.join(root, file)
                if ".git" in file_path:
                    continue

                # Get relative path from repo root
                file_path_key = os.path.relpath(file_path, self.repo_path)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        files[file_path_key] = content
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

        # order keys alphabetically
        files = dict(sorted(files.items()))
        return files

    @staticmethod
    def mb_to_bytes(mb: float) -> int:
        return int(mb * 1024 * 1024)

    def get_files_by_extension(
        self, extensions: list[str] | None = None
    ) -> dict[str, str]:
        if extensions is None:
            # Note: ipynb files are converted to .py during cleanup
            extensions = [
                ".c",
                ".cc",
                ".cpp",
                ".cu",
                ".h",
                ".hpp",
                ".java",
                ".jl",
                ".m",
                ".matlab",
                ".Makefile",
                ".md",
                ".pl",
                ".ps1",
                ".py",
                ".r",
                ".sh",
                "config.txt",
                ".rs",
                "readme.txt",
                "requirements_dev.txt",
                "requirements-dev.txt",
                "requirements.dev.txt",
                "requirements.txt",
                ".scala",
                ".yaml",
                ".yml",
            ]
        return {
            k: v for k, v in self.files.items() if k.lower().endswith(tuple(extensions))
        }

    def get_files_by_path(self, file_paths: list[str]) -> dict[str, str]:
        return {k: v for k, v in self.files.items() if k in file_paths}

    def get_repo_tree(self):
        repo_tree = ""
        for root, dirs, files in os.walk(self.repo_path):
            # Exclude the .git directory from the tree
            if ".git" in dirs:
                dirs.remove(".git")

            level = root.replace(self.repo_path, "").count(os.sep)
            indent = "│   " * (level - 1) + "├── " if level > 0 else ""

            # Don't print the starting path itself, just its contents
            if level > 0:
                repo_tree += f"{indent}{os.path.basename(root)}/\n"

            sub_indent = "│   " * level + "├── "
            for f in files:
                repo_tree += f"{sub_indent}{f}\n"
        return repo_tree

    def get_code_prompt(
        self,
        file_extensions: list[str] | None = None,
        token_counter: Callable | None = None,
        max_tokens: int | None = None,
        code_changes: list[dict[str, str]] | None = None,
    ) -> str:
        code_prompt = "Repo tree:\n" + self.get_repo_tree() + "\n\n"
        tokens = token_counter(code_prompt) if token_counter is not None else 0

        files_to_replace = {}
        if code_changes:
            files_to_replace = {
                cc["file_name"]: cc["discrepancy_code"] for cc in code_changes
            }
            logger.debug(
                f"Files to replace: {len(files_to_replace)}: {files_to_replace.keys()}"
            )

        for file_path, file_content in self.get_files_by_extension(
            file_extensions
        ).items():
            if file_path in files_to_replace:
                # Replace the original code with the changed code
                # if code_changes is provided
                logger.debug(f"Replacing code for {file_path} with changed code")
                file_content = files_to_replace[file_path]
            code_file = f"# ---\n# File: {file_path}\n# Content:\n{file_content}\n"
            if token_counter is not None:
                logger.debug(f"Adding file: {file_path}")
                num_tokens = token_counter(code_file)
                tokens += num_tokens
                logger.debug(
                    f"Number of tokens in file: {num_tokens}. "
                    f"Total number of tokens in code prompt: {tokens}"
                )
            if max_tokens and tokens > max_tokens:
                logger.warning(
                    f"Truncating. Max tokens reached for {self.github_url}. "
                    f"Max tokens for code is {max_tokens}"
                )
                break
            code_prompt += code_file
        return code_prompt


if __name__ == "__main__":
    raw_repo_dir = "data/test-repos-raw"
    os.makedirs(raw_repo_dir, exist_ok=True)
    max_file_size_mb = 1.0
    while True:
        github_url = input("Enter the GitHub URL: ")
        target_date_input = input("Enter the target date (YYYY-MM-DD): ")
        # Parse and reformat to ensure zero-padding
        try:
            parsed_date = datetime.strptime(target_date_input, "%Y-%m-%d")
            target_date = parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError as e:
            print(f"Invalid date format: {e}")
            continue
        loader = CodeLoader(github_url, max_file_size_mb, raw_repo_dir, target_date)
        prompt = loader.get_code_prompt()
        prompt_path = os.path.join(
            raw_repo_dir, loader.github_url_to_repo_name, "code_prompt.txt"
        )
        with open(prompt_path, "w") as f:
            f.write(prompt)
        print(f"Prompt saved to {prompt_path}")
        if input("Do you want to continue? (y/n): ").lower() == "n":
            break
