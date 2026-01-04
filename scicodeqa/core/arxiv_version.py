"""ArXiv version management for retrieving papers at specific dates."""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class ArxivVersionManager:
    """Manages arXiv paper versions and retrieves papers at specific dates."""

    # Class variable to track last API call time for rate limiting
    _last_api_call = 0

    @staticmethod
    def _rate_limited_get(url: str, min_interval: float = 3.0) -> requests.Response:
        """
        Make a rate-limited GET request to respect arXiv's API guidelines.

        Args:
            url: The URL to request
            min_interval: Minimum seconds between requests (default 3.0)

        Returns:
            Response object
        """
        current_time = time.time()
        time_since_last = current_time - ArxivVersionManager._last_api_call

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        response = requests.get(url)
        ArxivVersionManager._last_api_call = time.time()

        return response

    @staticmethod
    def extract_arxiv_id(url_or_id: str) -> str:
        """
        Extract arXiv ID from a URL or return the ID if already in ID format.

        Examples:
            https://arxiv.org/abs/2006.12834 -> 2006.12834
            https://arxiv.org/pdf/2006.12834v2.pdf -> 2006.12834
            2006.12834 -> 2006.12834
        """
        # If it's already an ID (contains dots and numbers)
        if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", url_or_id):
            # Remove version suffix if present
            return re.sub(r"v\d+$", "", url_or_id)

        # Extract from URL
        parsed = urlparse(url_or_id)
        path = parsed.path

        # Remove /abs/ or /pdf/ prefix
        path = path.replace("/abs/", "").replace("/pdf/", "")

        # Remove .pdf suffix and version suffix
        path = path.replace(".pdf", "")
        path = re.sub(r"v\d+$", "", path)

        return path.strip("/")

    @staticmethod
    def get_version_at_date(
        arxiv_url_or_id: str,
        target_date: datetime | date | str,
    ) -> str:
        """
        Get the PDF URL for the arXiv paper version that was available at a specific
        date.

        Args:
            arxiv_url_or_id: arXiv URL or ID (e.g., "https://arxiv.org/abs/2006.12834"
                or "2006.12834")
            target_date: Target date as datetime object, date object, or ISO string
                (e.g., "2020-07-01")

        Returns:
            PDF URL for the version available at target_date
                (e.g., "https://arxiv.org/pdf/2006.12834v2.pdf")
        """
        # Parse target date
        if isinstance(target_date, str):
            target_date = datetime.fromisoformat(target_date.replace("Z", "+00:00"))
        elif isinstance(target_date, date) and not isinstance(target_date, datetime):
            # Convert date to datetime at midnight UTC
            target_date = datetime.combine(target_date, datetime.min.time())

        # Make target_date timezone-aware if it isn't
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=timezone.utc)

        # Extract arXiv ID
        arxiv_id = ArxivVersionManager.extract_arxiv_id(arxiv_url_or_id)

        # Use the arXiv API directly to get version history from the Atom feed
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = ArxivVersionManager._rate_limited_get(api_url)
        response.raise_for_status()

        # Parse the XML response
        root = ET.fromstring(response.content)

        # Define namespace
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        # Find the entry (paper)
        entry = root.find("atom:entry", ns)
        if entry is None:
            raise ValueError(f"arXiv paper not found: {arxiv_id}")

        # Get published date (v1 date)
        published_elem = entry.find("atom:published", ns)
        if published_elem is None or published_elem.text is None:
            raise ValueError(
                f"Could not find published date for arXiv paper: {arxiv_id}"
            )
        published_str = published_elem.text
        published_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))

        # Get updated date (latest version date)
        updated_elem = entry.find("atom:updated", ns)
        if updated_elem is None or updated_elem.text is None:
            raise ValueError(f"Could not find updated date for arXiv paper: {arxiv_id}")
        updated_str = updated_elem.text
        updated_date = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))

        # Collect all version dates
        version_dates = [(1, published_date)]

        # If there are multiple versions, we need to query each one
        # The Atom feed doesn't include full version history in a single request
        if updated_date > published_date:
            # There are multiple versions, query each one
            version = 2
            while True:
                version_api_url = (
                    f"http://export.arxiv.org/api/query?id_list={arxiv_id}v{version}"
                )
                version_response = ArxivVersionManager._rate_limited_get(
                    version_api_url
                )
                version_response.raise_for_status()

                version_root = ET.fromstring(version_response.content)
                version_entry = version_root.find("atom:entry", ns)

                if version_entry is None:
                    # No more versions
                    break

                # Get the updated date for this version with None check
                version_updated_elem = version_entry.find("atom:updated", ns)
                if version_updated_elem is None or version_updated_elem.text is None:
                    # Skip this version if we can't get the date
                    break
                version_updated_str = version_updated_elem.text
                version_date = datetime.fromisoformat(
                    version_updated_str.replace("Z", "+00:00")
                )

                logger.debug(f"Found version {version} at {version_date.isoformat()}")

                version_dates.append((version, version_date))
                version += 1

        # Find the latest version that was published before or on target_date
        selected_version = 1
        selected_version_date = published_date

        for version_num, version_date in version_dates:
            if version_date <= target_date:
                selected_version = version_num
                selected_version_date = version_date
            else:
                break

        # Construct PDF URL for the selected version
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}v{selected_version}.pdf"

        logger.info(
            f"Selected arXiv version {selected_version} "
            f"(dated {selected_version_date.isoformat()}) "
            f"for target date {target_date.isoformat()}"
        )

        return pdf_url

    @staticmethod
    def get_pdf_url_at_date(
        arxiv_url_or_id: str,
        target_date: datetime | date | str | None,
    ) -> str:
        """
        Get the PDF URL for an arXiv paper at a specific date.

        If target_date is None, returns the latest version URL.

        Args:
            arxiv_url_or_id: arXiv URL or ID
            target_date: Target date as datetime, date object, string, or None for
            latest version

        Returns:
            PDF URL for the appropriate version
        """
        if target_date is None:
            # Return latest version URL (without version suffix, arXiv redirects to
            # latest)
            arxiv_id = ArxivVersionManager.extract_arxiv_id(arxiv_url_or_id)
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Get specific version at date (returns URL directly)
        return ArxivVersionManager.get_version_at_date(arxiv_url_or_id, target_date)
