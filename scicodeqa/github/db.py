from pathlib import Path

from tinydb import Query, TinyDB


class GitHubDB:
    def __init__(self, db_path: Path):
        self.db = TinyDB(db_path)

    def get_repo_by_id(self, id: int):
        return self.db.table("repos").search(Query().id == id)[0]

    def get_issue_by_id(self, id: int):
        return self.db.table("issues").search(Query().id == id)[0]

    def get_issues_by_ids(self, ids: list[int]):
        _issues = self.db.table("issues").search(
            Query().id.test(lambda _id: _id in ids)
        )
        # sort by same order as ids
        _issues = sorted(_issues, key=lambda x: ids.index(x["id"]))
        return _issues

    def get_comment_by_id(self, id: int):
        return self.db.table("comments").search(Query().id == id)[0]

    def get_repo_issues(self, repo_id: int):
        return self.db.table("issues").search(Query().repository_id == repo_id)

    def get_comments_by_issue_id(self, issue_id: int):
        comments = self.db.table("comments").search(Query().issue_id == issue_id)
        return sorted(comments, key=lambda x: x.get("created_at", ""))

    def get_all_issues(self):
        """Get all issues from the database."""
        return self.db.table("issues").all()

    def get_repos_by_ids(self, repo_ids: list[int]) -> dict[int, dict]:
        """Get multiple repositories by their IDs."""
        if not repo_ids:
            return {}
        repos = self.db.table("repos").search(
            Query().id.test(lambda _id: _id in repo_ids)
        )
        return {repo["id"]: repo for repo in repos}

    def get_comments_by_issue_ids(self, issue_ids: list[int]) -> dict[int, list[dict]]:
        """Get all comments for multiple issues at once."""
        if not issue_ids:
            return {}
        comments = self.db.table("comments").search(
            Query().issue_id.test(lambda _id: _id in issue_ids)
        )
        # Group comments by issue_id
        comments_by_issue = {}
        for comment in comments:
            issue_id = comment["issue_id"]
            if issue_id not in comments_by_issue:
                comments_by_issue[issue_id] = []
            comments_by_issue[issue_id].append(comment)

        # Sort comments by created_at for each issue
        for issue_id in comments_by_issue:
            comments_by_issue[issue_id] = sorted(
                comments_by_issue[issue_id], key=lambda x: x.get("created_at", "")
            )

        return comments_by_issue

    def get_issues_with_related_data(
        self, issues: list[dict]
    ) -> tuple[list[dict], dict[int, dict], dict[int, list[dict]]]:
        """Get issues with all related data (repos, comments) in bulk."""
        if not issues:
            return issues, {}, {}

        # Extract unique repo_ids and issue_ids
        repo_ids = list(
            set(
                issue.get("repository_id")
                for issue in issues
                if issue.get("repository_id")
            )
        )
        issue_ids = [issue.get("id") for issue in issues if issue.get("id")]

        # Bulk fetch related data
        repos = self.get_repos_by_ids(repo_ids)
        comments_by_issue = self.get_comments_by_issue_ids(issue_ids)

        return issues, repos, comments_by_issue

    def get_issues_with_comments(self, issue_ids: list[int]) -> set[int]:
        """Get issue IDs that have at least one comment."""
        if not issue_ids:
            return set()

        # Get all comments for these issues
        comments = self.db.table("comments").search(
            Query().issue_id.test(lambda _id: _id in issue_ids)
        )

        # Return set of issue IDs that have comments
        return set(comment["issue_id"] for comment in comments)

    def get_contributors(self, repository_id: int) -> list[str]:
        """Get all contributors for a repository."""
        repo = self.get_repo_by_id(repository_id)
        contributors = []
        for contributor in repo.get("contributors", []):
            if contributor.get("login"):
                contributors.append(contributor.get("login"))

        return contributors
