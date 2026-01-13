import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from logging.config import fileConfig
from pathlib import Path

import requests
from mistralai import Mistral

from scicoqa.core.arxiv_version import ArxivVersionManager

fileConfig("logging.ini")
logger = logging.getLogger(__name__)

# HACK: map DOI to PDF URL
DOI2PDF_URL = {
    "https://doi.org/10.1038/s41597-023-02020-6": "https://www.nature.com/articles/s41597-023-02020-6.pdf",  # noqa: E501
    "https://doi.org/10.1093/jamia/ocaa139": "https://watermark02.silverchair.com/ocaa139.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA1cwggNTBgkqhkiG9w0BBwagggNEMIIDQAIBADCCAzkGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMoNmFjpgZqX8Bpnl7AgEQgIIDCs6s88umhUiy4A2x0bGTM6Sa_fSnE7lawymivICtkafW0cn-lvxzM7tUFrnvvLVtrNnl3nP5HpDiua5Ow5lFpVWX4Zbzjnes8s2m_cWJx9Hx2EEQTsvDDozgQiC7_ENuNLxEGzjGnC-syKlTDGagvTT3wOhUJrPpt99xrtyxNS4s1rkpGn_WM8Sx1LK0XSjJTEPzVQ49wjXgiT3znte0sNfYVgdmIcnwLBC0wJV3Z70jCVcFZjVRfHXShRNmBjryP3YVn-Xcn0-D-YxvRKfhaw1UBYaIweSXWE6L2DS38HForX663_XlYJjx4VVRXxoCe-6W1G6Dz8Ez4sJyLKqXKKTq0ywxsvGiuHooZUzE4qCB3NBCrbbHHunl8k4BApGy0sKlXeo4RtVE2F2Ua7Y6mO3XJ8eTuVQ64k_fiKfO_n70POMqZLw8d2l8VIBhGSyjDsoGSEKsdx1VRH4I9oPVwVRec3KQDB_P477_Wj3TLQeyaSNuhtT4ptrFb3Vfu25LWTrucu3NdD__FG1k2uHAIZQRVKGsMrq-Js9cIUINc5zTyp6pO93adCARwe1jcL5VawwNeZngAL6gOPVZiH52NDLpeCwybX5hPyiy0SPfANqL2fkLL_BzZu1sOi3Oh1kZYqTtJzbngpqBiYcpsP1gOyQZf52EuCuKwUyTACisRzo1-Aji_ZGhZeliKQXmviAxgtKJQ2U334dBo9J4lnlK5GN-ldtRH5OG8HQvmKx6gS658xqy45c7sR8WcBK3MbugVDTjeLIZ1Vbg_3uOm9RuEfiDxQlbORj_xfry4_FuUqCViCCYvbHHckaXH1lN3M5h0EJlvOJRJWFN9_724dV368OwTAcGwbsn_1j063RwIFBY66X7HGs2mAv5uYId5ZUoONU6pyKYCQF6UBgiQE5uRmup4bnwMcGPtSjF-0awMoE6wZNqynBW8qtnpvYHvsLwVbXE7r5oIV9V-VV6MB0TfJeVdhRjuy9VP8KBo5-P95ZbWhzAlHSX36D9ljlMIY2FMFo2xXm8DQ__FCY",  # noqa: E501
}


class MistralOCR:
    def __init__(
        self,
        api_key: str | None = None,
        output_dir: Path = Path("data") / "papers",
    ):
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")
        if api_key is None:
            raise ValueError("MISTRAL_API_KEY is not set")
        self.client = Mistral(api_key=api_key)
        self.output_dir = output_dir

    def _get_paper_paths(self, paper_id: str) -> tuple[Path, Path, Path, Path]:
        """Get the file paths for a paper.

        Returns:
            tuple containing:
            - path to filtered markdown (no references)
            - path to full markdown (with references)
            - path to metadata
            - path to paper directory
        """
        paper_dir = self.output_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        return (
            paper_dir / "paper.md",
            paper_dir / "paper_raw.md",
            paper_dir / "metadata.json",
        )

    def _load_paper(self, paper_id: str) -> str | None:
        """Load processed paper if available.

        Returns the filtered version (without references) if available.
        """
        filtered_path, full_path, meta_path = self._get_paper_paths(paper_id)

        # Check if both versions and metadata exist
        if filtered_path.exists() and full_path.exists() and meta_path.exists():
            with open(filtered_path, "r") as f:
                text = f.read()
            logger.info(f"Loaded paper {paper_id} from {filtered_path}")
            return text

        return None

    def save(self, paper_id: str, raw_text: str, processed_text: str, url: str):
        """Save processed paper and metadata."""
        paper_dir = self.output_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Save full text with references
        full_path = paper_dir / "paper_raw.md"
        with open(full_path, "w") as f:
            f.write(raw_text)

        processed_path = paper_dir / "paper.md"
        with open(processed_path, "w") as f:
            f.write(processed_text)

        # Save metadata
        metadata = {
            "paper_id": paper_id,
            "source_url": url,
            "processed_at": datetime.now().isoformat(),
        }
        meta_path = paper_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _download_pdf(self, document_url: str) -> Path:
        """Download PDF from URL to a temporary file."""
        logger.debug(f"Downloading PDF from {document_url}")
        response = requests.get(document_url, stream=True)
        response.raise_for_status()

        # Create temporary file with .pdf extension
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_path = Path(temp_file.name)

        # Write content to file
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()

        logger.debug(f"PDF downloaded to {temp_path}")
        return temp_path

    def _process_with_local_upload(
        self, document_url: str | None = None, local_file_path: Path | None = None
    ) -> str:
        """Process PDF by downloading locally (or using existing local file) and
        uploading to Mistral API."""
        temp_path = None
        should_cleanup = False
        try:
            # Use local file if provided, otherwise download
            if local_file_path:
                pdf_path = local_file_path
                logger.debug(f"Using local PDF file: {pdf_path}")
            elif document_url:
                # Download PDF locally
                pdf_path = self._download_pdf(document_url)
                temp_path = pdf_path
                should_cleanup = True
            else:
                raise ValueError(
                    "Either document_url or local_file_path must be provided"
                )

            # Upload file to Mistral API
            with open(pdf_path, "rb") as pdf_file:
                uploaded_pdf = self.client.files.upload(
                    file={
                        "file_name": f"uploaded_{pdf_path.name}",
                        "content": pdf_file,
                    },
                    purpose="ocr",
                )

            # Get signed URL for the uploaded file
            signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)

            # Process OCR using the signed URL
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                include_image_base64=False,
            )

            # Process pages
            pages = []
            for page in ocr_response.pages:
                pages.append(page.markdown)
            markdown = "\n\n".join(pages)

            return markdown

        finally:
            # Clean up temporary file only if we downloaded it
            if should_cleanup and temp_path and temp_path.exists():
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file {temp_path}")

    def _filter_markdown(self, markdown: str) -> str:
        """Filter markdown to remove references section and other unwanted content."""
        lines = markdown.split("\n")
        filtered_lines = []
        skip_mode = False

        for line in lines:
            # Skip references section
            if line.startswith("#") and line.lower().strip().endswith("# references"):
                skip_mode = True
                continue
            elif skip_mode and line.startswith("#"):
                skip_mode = False

            # Skip empty lines at the start
            if not filtered_lines and not line.strip():
                continue

            if not skip_mode:
                filtered_lines.append(line)

        # Remove trailing empty lines
        while filtered_lines and not filtered_lines[-1].strip():
            filtered_lines.pop()

        return "\n".join(filtered_lines)

    def _extract_paper_id(self, document_url: str) -> str:
        """Extract paper ID from arXiv URL."""
        logger.info(f"Extracting paper ID from URL: {document_url}")
        if "arxiv.org" in document_url:
            parts = document_url.split("/")
            paper_id = parts[-1].replace(".pdf", "")
            logger.info(f"Extracted arXiv ID: {paper_id}")
            return paper_id
        elif "openreview.net" in document_url:
            i = document_url.index("id=")
            paper_id = document_url[i + 3 :]
            logger.info(f"Extracted OpenReview ID: {paper_id}")
            return paper_id
        elif "biorxiv.org" in document_url:
            document_url = document_url.replace(".pdf", "")
            document_url = document_url.replace(".full", "")
            paper_id = "/".join(document_url.split("/")[:-2])
            logger.info(f"Extracted BiorXiv ID: {paper_id}")
            return paper_id
        elif "dl.acm.org" in document_url:
            # Extract DOI from ACM URL
            # URL format: https://dl.acm.org/doi/pdf/10.1145/3726302.3730337
            # Extract the DOI parts after 10.1145/
            parts = document_url.split("/")
            doi_parts = [p for p in parts if "." in p and p[0].isdigit()]
            if doi_parts:
                paper_id = doi_parts[-1].replace(".pdf", "")
                logger.info(f"Extracted ACM ID: {paper_id}")
                return paper_id
            # Fallback to hash if DOI extraction fails

            paper_id = hashlib.sha256(document_url.encode()).hexdigest()[:16]
            logger.info(f"Generated hash ID for ACM paper: {paper_id}")
            return paper_id
        else:
            # For non-arXiv URLs, use a hash of the URL
            paper_id = hashlib.sha256(document_url.encode()).hexdigest()[:16]
            logger.info(f"Generated hash ID: {paper_id}")
            return paper_id

    def __call__(
        self, document_url: str, target_date: str | datetime | None = None
    ) -> tuple[str, str]:
        """Process a paper URL and return its text content.

        The function will:
        1. Check if the paper is already processed
        2. If not, download and process it (or use local file for ACM papers)
        3. Filter out references and clean up the text
        4. Save as markdown with metadata

        Args:
            document_url: URL to the paper (arXiv, ACM, OpenReview, etc.)
            target_date: Optional date to retrieve a specific arXiv version.
                        For arXiv papers, if provided, will retrieve the version
                        available at that date. Ignored for non-arXiv papers.
        """
        logger.debug(f"MistralClient({document_url}, target_date={target_date})")

        if document_url in DOI2PDF_URL:
            document_url = DOI2PDF_URL[document_url]

        # For arXiv papers with target_date, get the version-specific URL
        original_url = document_url
        is_arxiv_url = "arxiv.org" in document_url
        ends_with_version = bool(re.search(r"v\d+(\.pdf)?$", document_url))
        if is_arxiv_url and ends_with_version:
            logger.info(f"Using hardcoded version arXiv URL: {document_url}")
        elif target_date is not None and is_arxiv_url and not ends_with_version:
            try:
                version_url = ArxivVersionManager.get_pdf_url_at_date(
                    document_url, target_date
                )
                logger.info(f"Using version-specific arXiv URL: {version_url}")
                document_url = version_url
            except Exception as e:
                logger.warning(
                    f"Failed to get arXiv version at date {target_date}: {e}. "
                    f"Using original URL: {document_url}"
                )
                raise e

        # Extract paper ID and check if already processed
        paper_id = self._extract_paper_id(original_url)

        # If we have a target_date, include it in the paper_id to cache different
        # versions
        if target_date is not None and "arxiv.org" in original_url:
            # Add target_date to paper_id for version-specific caching
            if isinstance(target_date, datetime):
                date_str = target_date.isoformat()
            else:
                date_str = str(target_date)
            paper_id = f"{paper_id}_{date_str[:10]}"  # Use YYYY-MM-DD

        existing_text = self._load_paper(paper_id)
        if existing_text is not None:
            return existing_text, document_url

        # Handle ACM papers from local directory
        if "dl.acm.org" in document_url:
            # Extract DOI suffix (e.g., 3726302.3730337 from 10.1145/3726302.3730337)
            parts = document_url.split("/")
            doi_parts = [p for p in parts if "." in p and p[0].isdigit()]
            if doi_parts:
                doi_suffix = doi_parts[-1].replace(".pdf", "")
                local_pdf_path = Path("data") / "acm-pdfs" / f"{doi_suffix}.pdf"
                if local_pdf_path.exists():
                    logger.info(f"Using local ACM PDF: {local_pdf_path}")
                    raw_text = self._process_with_local_upload(
                        local_file_path=local_pdf_path
                    )
                else:
                    logger.warning(f"Local ACM PDF not found: {local_pdf_path}")
                    raise FileNotFoundError(f"ACM PDF not found at {local_pdf_path}")
            else:
                raise ValueError(f"Could not extract DOI from ACM URL: {document_url}")
        else:
            # Convert arXiv abstract URL to PDF URL if needed
            if "arxiv.org/" in document_url and document_url.endswith(".pdf"):
                document_url = document_url.strip()[:-4]
            if "arxiv.org/abs" in document_url:
                document_url = document_url.replace("arxiv.org/abs", "arxiv.org/pdf")
            elif "openreview.net" in document_url:
                document_url = document_url.replace(
                    "openreview.net/forum", "openreview.net/pdf"
                )
            elif "biorxiv.org" in document_url:
                if not document_url.endswith(".full.pdf"):
                    document_url += ".full.pdf"

            raw_text = self._process_with_local_upload(document_url=document_url)

        processed_text = self._filter_markdown(raw_text)

        self.save(paper_id, raw_text, processed_text, document_url)

        return processed_text, document_url
