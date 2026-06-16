"""
Automated FITS file downloader from ISSDC PRADAN portal.

Handles Keycloak SSO login, data browsing, and file download.
Supports SoLEXS and HEL1OS Level-1 data from Aditya-L1.

PRADAN URL patterns:
    Browse: https://pradan.issdc.gov.in/al1/protected/browse.xhtml?id=solexs
    Download: https://pradan.issdc.gov.in/al1/protected/downloadData/{payload}/level1/{year}/{month}/{day}/{filename}?{payload}
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

AL1_BASE = "https://pradan.issdc.gov.in/al1"


class PRADANDownloader:
    """Downloads FITS files from ISSDC PRADAN portal with Keycloak SSO auth."""

    def __init__(
        self,
        username: str,
        password: str,
        download_dir: str = "data/pradan",
    ):
        self.username = username
        self.password = password
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        self.logged_in = False

    def login(self) -> bool:
        """Authenticate via Keycloak SSO."""
        logger.info(f"Logging in as {self.username}...")

        # GET protected page -> redirect to Keycloak
        resp = self.session.get(
            f"{AL1_BASE}/protected/payload.xhtml",
            allow_redirects=True,
            timeout=30,
        )

        # Parse Keycloak login form
        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.find("form", id="kc-form-login")

        if not form:
            # Check if already logged in
            if "payload" in resp.url or "pradan" in resp.url:
                logger.info("Already logged in")
                self.logged_in = True
                return True
            logger.error("Could not find login form")
            return False

        form_action = form.get("action", "")
        payload = {}
        for inp in form.find_all("input"):
            name = inp.get("name")
            value = inp.get("value", "")
            if name:
                payload[name] = value

        payload["username"] = self.username
        payload["password"] = self.password

        # POST credentials
        resp = self.session.post(form_action, data=payload, allow_redirects=True, timeout=30)

        # Check success
        if resp.status_code == 200 and "login" not in resp.url.lower():
            logger.info("Login successful!")
            self.logged_in = True
            return True

        logger.error(f"Login failed: {resp.status_code}")
        return False

    def _browse_page(self, payload: str) -> List[Dict]:
        """Browse the data page and extract file info."""
        if not self.logged_in:
            return []

        url = f"{AL1_BASE}/protected/browse.xhtml?id={payload}"
        resp = self.session.get(url, timeout=30)

        if resp.status_code != 200 or "login" in resp.url:
            logger.error(f"Failed to access {payload} browse page")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        files = []

        # Find data table rows
        for table in soup.find_all("table"):
            tbody = table.find("tbody")
            if not tbody:
                continue
            rows = tbody.find_all("tr")
            if len(rows) < 3:
                continue

            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                file_info = {}
                for cell in cells:
                    link = cell.find("a")
                    text = cell.get_text(strip=True)

                    if link:
                        href = link.get("href", "")
                        if href and "#" not in href:
                            if href.startswith("/"):
                                href = f"https://pradan.issdc.gov.in{href}"
                            file_info["url"] = href
                            file_info["filename"] = text or href.split("/")[-1].split("?")[0]

                    # Extract date from text
                    if text and re.match(r"\d{4}-\d{2}-\d{2}", text):
                        file_info["date"] = text

                    # Extract size
                    if text and re.match(r"[\d.]+", text):
                        try:
                            file_info["size_mb"] = float(text)
                        except ValueError:
                            pass

                if file_info.get("url") and file_info.get("filename"):
                    files.append(file_info)

        logger.info(f"Found {len(files)} files for {payload}")
        return files

    def list_solexs_files(self) -> List[Dict]:
        """List available SoLEXS files."""
        return self._browse_page("solexs")

    def list_hel1os_files(self) -> List[Dict]:
        """List available HEL1OS files."""
        return self._browse_page("hel1os")

    def download_file(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Download a single file."""
        if not self.logged_in:
            return None

        if not filename:
            filename = url.split("/")[-1].split("?")[0]

        local_path = self.download_dir / filename
        if local_path.exists():
            logger.info(f"Already exists: {filename}")
            return local_path

        logger.info(f"Downloading: {filename}")
        try:
            resp = self.session.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Saved: {filename} ({size_mb:.1f} MB)")
            return local_path
        except Exception as e:
            logger.error(f"  Failed: {e}")
            local_path.unlink(missing_ok=True)
            return None

    def download_latest(
        self,
        n_solexs: int = 1,
        n_hel1os: int = 1,
    ) -> Tuple[List[Path], List[Path]]:
        """Download the most recent SoLEXS and HEL1OS files."""
        solexs_paths = []
        hel1os_paths = []

        if n_solexs > 0:
            files = self.list_solexs_files()
            for f in files[:n_solexs]:
                path = self.download_file(f["url"], f["filename"])
                if path:
                    solexs_paths.append(path)
                time.sleep(1)

        if n_hel1os > 0:
            files = self.list_hel1os_files()
            for f in files[:n_hel1os]:
                path = self.download_file(f["url"], f["filename"])
                if path:
                    hel1os_paths.append(path)
                time.sleep(1)

        return solexs_paths, hel1os_paths

    def download_date(
        self,
        date: str,
        payload: str = "both",
    ) -> Tuple[List[Path], List[Path]]:
        """Download files for a specific date (YYYY-MM-DD)."""
        solexs_paths = []
        hel1os_paths = []

        if payload in ("solexs", "both"):
            files = self.list_solexs_files()
            for f in files:
                if date in f.get("date", "") or date in f.get("filename", ""):
                    path = self.download_file(f["url"], f["filename"])
                    if path:
                        solexs_paths.append(path)
                    break

        if payload in ("hel1os", "both"):
            files = self.list_hel1os_files()
            for f in files:
                if date in f.get("date", "") or date in f.get("filename", ""):
                    path = self.download_file(f["url"], f["filename"])
                    if path:
                        hel1os_paths.append(path)
                    break

        return solexs_paths, hel1os_paths


def download_from_pradan(
    username: str,
    password: str,
    download_dir: str = "data/pradan",
    n_solexs: int = 1,
    n_hel1os: int = 1,
) -> Tuple[List[Path], List[Path]]:
    """Convenience function to download from PRADAN."""
    downloader = PRADANDownloader(username, password, download_dir)
    if not downloader.login():
        return [], []
    return downloader.download_latest(n_solexs, n_hel1os)
