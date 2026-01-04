import logging
from logging.config import fileConfig

from scicodeqa.github.issue_img_to_text import ISSUE_IMAGE_TO_TEXT

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


def issue_to_text(
    issue: dict, truncate_after: int | None = None, replace_images: bool = False
) -> str:
    """Convert a GitHub issue dict to a text blob (title + body)."""
    title = issue.get("title", "")
    body = issue.get("body", "")
    text = ""
    if title:
        text += title + "\n"
    if body:
        text += body

    if replace_images:
        for img, replacement_text in ISSUE_IMAGE_TO_TEXT.items():
            if img in text:
                logger.info(
                    f"Replacing image {img[:10]}... with text {replacement_text[:10]}.."
                )
                text = text.replace(img, replacement_text)

    words = text.split()
    if truncate_after is not None and len(words) > truncate_after:
        text = " ".join(words[:truncate_after]) + "..."
    return text


def comments_to_text(
    comments: list[dict], contributors: list[str] | None = None
) -> str:
    """Convert a list of GitHub comments dict to a text blob."""
    comments = sorted(comments, key=lambda x: x.get("created_at", ""))
    text = ""
    for i, comment in enumerate(comments):
        prefix = f"#### Comment {i + 1}"
        comment_text = comment.get("body", "")
        if contributors:
            comment_author = comment.get("user_login", "Unknown")
            is_contributor = comment_author in contributors
            contributor_tag = "(contributor)" if is_contributor else ""
            prefix += f" {contributor_tag}"

        prefix += "\n"
        text += f"{prefix}{comment_text}\n"

    return text
