import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from logging.config import fileConfig
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from tinydb import Query, TinyDB
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from github import Auth, Github, Issue, IssueComment, Repository

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    search_str: str = "arxiv.org"
    qualifiers: list[str] = None
    out_database: Path | None = None
    filter_homepage_by: str = None
    crawl_issues: bool = True  # Whether to crawl issues and comments
    min_rate_limit: int = 150  # Minimum rate limit before sleeping
    rate_limit_buffer: int = 30  # Extra seconds to sleep beyond reset time
    seconds_between_requests: float = 1.0  # Delay between API requests
    max_workers: int = 8

    def __post_init__(self):
        # convert qualifiers to dict
        if self.qualifiers is not None:
            self.qualifiers = {
                q.split(":", 1)[0]: q.split(":", 1)[1] for q in self.qualifiers
            }
        else:
            self.qualifiers = {}

        if self.out_database is None:
            if self.crawl_issues:
                self.out_database = (
                    Path("out") / "data_collection" / "github_crawl" / "ric.json"
                )
            else:
                self.out_database = (
                    Path("out") / "data_collection" / "github_crawl" / "repos-only.json"
                )

        # check that out_database is a valid path and ends with .json
        if not self.out_database.name.endswith(".json"):
            raise ValueError("out_database must end with .json")
        os.makedirs(self.out_database.parent, exist_ok=True)


def check_and_handle_rate_limit(
    github_instance, min_remaining=150, sleep_buffer=30, check_core_api=False
):
    if check_core_api:
        # Get Core API rate limits specifically
        try:
            rate_limit_details = github_instance.get_rate_limit()
            current_remaining = rate_limit_details.raw_data["resources"]["core"][
                "remaining"
            ]
            max_limit = rate_limit_details.raw_data["resources"]["core"]["limit"]
            reset_time = rate_limit_details.raw_data["resources"]["core"]["reset"]
            api_type = "Core API"
        except Exception as e:
            logger.warning(
                f"Failed to get Core API rate limits, falling back to rate_limiting "
                f"property: {e}"
            )
            current_remaining, max_limit = github_instance.rate_limiting
            reset_time = github_instance.rate_limiting_resettime
            api_type = "Unknown API"
    else:
        # Use the simpler rate_limiting property for current status
        current_remaining, max_limit = github_instance.rate_limiting
        reset_time = github_instance.rate_limiting_resettime
        api_type = "Search API" if max_limit == 30 else "Core API"

    current_time = time.time()

    # Log rate limit status with type information for debugging
    logger.info(
        f"Rate limit status ({api_type}): {current_remaining}/{max_limit} remaining, "
        f"resets in {reset_time - current_time:.0f} seconds"
    )

    # Ensure we're comparing numbers by explicitly converting to int
    current_remaining = int(current_remaining)
    min_remaining = int(min_remaining)

    if current_remaining <= min_remaining:
        sleep_time = (reset_time - current_time) + sleep_buffer
        if sleep_time > 0:
            logger.warning(
                f"Rate limit low ({current_remaining} remaining). "
                f"Sleeping for {sleep_time:.0f} seconds until reset + buffer"
            )
            time.sleep(sleep_time)
        else:
            logger.info("Rate limit reset time has already passed, continuing...")

    return current_remaining


def repo_repr(r: Repository) -> dict:
    # get contributors
    contributors = r.get_contributors()
    contributors = [
        {
            "id": c.id,
            "login": c.login,
            "contributions": c.contributions,
        }
        for c in contributors
    ]

    return {
        "contributors": contributors,
        "created_at": r.created_at.isoformat(),
        "description": r.description,
        "html_url": r.html_url,
        "homepage": r.homepage if r.homepage else None,
        "id": r.id,
        "language": r.language,
        "license": r.license.name if r.license else None,
        "name": r.full_name,
        "pushed_at": r.pushed_at.isoformat(),
        "stargazers_count": r.stargazers_count,
        "topics": r.topics,
        "updated_at": r.updated_at.isoformat(),
        "url": r.url,
        "user_id": r.owner.id,
        "user_login": r.owner.login,
    }


def issue_repr(i: Issue, repository: Repository) -> dict:
    return {
        "body_html": i.body_html,
        "body_text": i.body_text,
        "body": i.body,
        "created_at": i.created_at.isoformat(),
        "html_url": i.html_url,
        "id": i.id,
        "labels": [label.name for label in i.labels],
        "number": i.number,
        "repository_id": repository.id,
        "state": i.state,
        "title": i.title,
        "updated_at": i.updated_at.isoformat(),
        "url": i.url,
        "user_id": i.user.id,
        "user_login": i.user.login,
    }


def comment_repr(c: IssueComment, issue: Issue) -> dict:
    return {
        "body_html": c.body_html,
        "body_text": c.body_text,
        "body": c.body,
        "created_at": c.created_at.isoformat(),
        "html_url": c.html_url,
        "id": c.id,
        "issue_id": issue.id,
        "updated_at": c.updated_at.isoformat(),
        "url": c.url,
        "user_id": c.user.id,
        "user_login": c.user.login,
    }


def process_issue_comments(issue: Issue, comments_table, db_lock, github_instance=None):
    """Process all comments for a single issue and return comment for batch insert."""
    comment_results = []
    try:
        # Use the provided authenticated github instance if available
        if github_instance is not None:
            # Get the issue from the authenticated instance to ensure proper auth
            repo = github_instance.get_repo(issue.repository.full_name)
            authenticated_issue = repo.get_issue(issue.number)
            comments = authenticated_issue.get_comments()
        else:
            # Fallback to the original method
            comments = issue.get_comments()

        for c in comments:
            try:
                comment_data = comment_repr(c, issue)
                comment_results.append(comment_data)
            except Exception as e:
                logger.error(f"Error processing comment {c.url}: {e}")
    except Exception as e:
        logger.error(f"Error fetching comments for issue {issue.url}: {e}")

    return comment_results


def main(args: Args):
    load_dotenv()
    g = Github(
        auth=Auth.Token(os.getenv("GITHUB_TOKEN")),
        per_page=100,
        seconds_between_requests=args.seconds_between_requests,
        timeout=30,
        retry=3,
    )
    DB = TinyDB(
        args.out_database,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )
    REPOS = DB.table("repos")
    ISSUES = DB.table("issues")
    COMMENTS = DB.table("comments")

    # Create a lock for thread-safe database operations
    db_lock = threading.Lock()

    # Check initial rate limit status
    logger.info("Checking initial rate limit status...")
    check_and_handle_rate_limit(
        g, min_remaining=args.min_rate_limit, sleep_buffer=args.rate_limit_buffer
    )

    repos_search = g.search_repositories(args.search_str, **args.qualifiers)
    num_search_results = repos_search.totalCount
    # get all repo data from database for efficient lookups
    repos_db = REPOS.all()
    repos_db_by_url = {r["url"]: r for r in repos_db}

    # Pre-load all existing issue IDs grouped by repository for efficient lookups
    existing_issues_by_repo = {}
    if args.crawl_issues:
        all_issues = ISSUES.all()
        for issue in all_issues:
            repo_id = issue["repository_id"]
            if repo_id not in existing_issues_by_repo:
                existing_issues_by_repo[repo_id] = set()
            existing_issues_by_repo[repo_id].add(issue["id"])

    repos = []
    for r in tqdm(
        repos_search, desc="Filtering repositories", total=num_search_results
    ):
        if r.url in repos_db_by_url:
            # Skip repos that are already fully processed (have issues data)
            # Only process if this is a new repo or if we need to update it
            existing_repo = repos_db_by_url[r.url]

            # If we're not crawling issues, skip repos that already exist
            if not args.crawl_issues:
                logger.debug(
                    f"Skipping already processed repo (crawl_issues=F): {r.full_name}"
                )
                continue

            if r.id in existing_issues_by_repo and existing_issues_by_repo[r.id]:
                logger.debug(f"Skipping already processed repo: {r.full_name}")
                continue
            # If repo exists but has no issues, we'll reprocess it

        if args.filter_homepage_by is not None:
            in_homepage = args.filter_homepage_by in (r.homepage or "")
            in_description = args.filter_homepage_by in (r.description or "")
            if not in_homepage and not in_description:
                continue
        repos.append(r)
    logger.info(f"Found {len(repos)} repositories (filtered from {num_search_results})")

    pbar_repos = tqdm(desc="Repositories", total=len(repos))
    for repo_idx, r in enumerate(repos):
        logger.debug(f"Repository: {r.url}")
        pbar_repos.set_postfix({"repo": r.full_name})

        # Check rate limit every 10 repositories
        if repo_idx % 10 == 0:
            remaining = check_and_handle_rate_limit(
                g,
                min_remaining=args.min_rate_limit,
                sleep_buffer=args.rate_limit_buffer,
                check_core_api=True,
            )
            logger.debug(f"Rate limit check at repo {repo_idx}: {remaining} remaining")

        # Only process issues and comments if crawl_issues is True
        if args.crawl_issues:
            existing_issue_ids = existing_issues_by_repo.get(r.id, set())
            if existing_issue_ids and r.url in repos_db_by_url:
                existing_repo = repos_db_by_url[r.url]
                if existing_repo.get("updated_at") == r.updated_at.isoformat():
                    logger.debug(
                        f"Repo not updated since last fetch, skipping: {r.full_name}"
                    )
                    pbar_repos.update(1)
                    continue

            issues = r.get_issues()
            issues = [
                i
                for i in issues
                if i.pull_request is None and i.id not in existing_issue_ids
            ]

            # Skip if no new issues to process
            if not issues:
                logger.debug(f"No (new) issues to process for {r.full_name}")
                pbar_repos.update(1)
                continue

            pbar_issues = tqdm(desc="Issues", leave=False, total=len(issues))

            # Check rate limit before processing issues
            # (comment fetching is API intensive)
            check_and_handle_rate_limit(
                g,
                min_remaining=args.min_rate_limit,
                sleep_buffer=args.rate_limit_buffer,
                check_core_api=True,
            )

            # Process comments concurrently using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit all comment processing tasks
                comment_futures = {
                    executor.submit(
                        process_issue_comments, issue, COMMENTS, db_lock
                    ): issue
                    for issue in issues
                }

                # Collect all results for batch processing
                all_comments = []
                issues_to_insert = []

                # Process completed comment tasks
                for future in as_completed(comment_futures):
                    issue = comment_futures[future]
                    try:
                        comment_results = future.result()
                        all_comments.extend(comment_results)
                        issues_to_insert.append(issue_repr(issue, r))
                        logger.debug(
                            f"Issue: {issue.title:.10s}... ({len(comment_results)} "
                            f"comments)"
                        )
                    except Exception as e:
                        logger.error(f"Error processing issue {issue.url}: {e}")

                    pbar_issues.update(1)

                # Batch insert all comments and issues
                with db_lock:
                    if all_comments:
                        COMMENTS.insert_multiple(all_comments)
                    if issues_to_insert:
                        ISSUES.insert_multiple(issues_to_insert)

            pbar_issues.close()

        try:
            repo_dict = repo_repr(r)
            repo_dict["search_str"] = args.search_str

            # Check if repo already exists and update/insert accordingly
            if r.url in repos_db_by_url:
                REPOS.update(repo_dict, Query().url == r.url)
                logger.debug(f"Updated existing repository: {r.full_name}")
            else:
                REPOS.insert(repo_dict)
                logger.debug(f"Inserted new repository: {r.full_name}")
        except Exception as e:
            logger.error(f"Error inserting/updating repository {r.url}: {e}")
            # delete the issues and comments from the database
            ISSUES.remove(Query().repository_id == r.id)
            COMMENTS.remove(Query().repository_id == r.id)
        pbar_repos.update(1)
    pbar_repos.close()


if __name__ == "__main__":
    with logging_redirect_tqdm(loggers=[logging.root]):
        args, unknown = simple_parsing.parse_known_args(Args)
        logger.info(f"Args: {args}")
        if unknown:
            logger.warning(f"Unknown arguments: {unknown}")

        main(args)
