#!/usr/bin/env python3
"""Download the raw Norman et al. 2019 Perturb-seq dataset into ``data/raw/``.

Source selection (both checked 2026-09-10):

* GEO GSE133344 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344)
  provides a 1.1 GB gzipped MTX matrix plus barcode/gene/identity sidecar
  files (and a 10 GB raw tar). GEO does not state an explicit license.
* scPerturb (https://scperturb.org) mirrors the same dataset, harmonized, as a
  single ``.h5ad`` on Zenodo record 7041849 under an explicit **CC-BY-4.0**
  license.

Because the scPerturb mirror has a clear license, a directly loadable AnnData
file, and a published MD5 checksum, it is used as the primary source. The small
GEO per-cell guide-identity CSV is also fetched for cross-referencing.

Accession, source URL and download date are recorded in ``configs/data.yaml``.
Re-running is safe: existing files with a matching MD5 are skipped unless
``--force`` is given.

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --force
    python scripts/download_data.py --config configs/data.yaml --skip-guide-annotations
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"

# Fallback values if the config does not define them (checked 2026-09-10).
DEFAULT_URL = (
    "https://zenodo.org/records/7041849/files/"
    "NormanWeissman2019_filtered.h5ad?download=1"
)
DEFAULT_MD5 = "c870e6967d91c017d9da827bab183cd6"
DEFAULT_RAW_FILE = "data/raw/NormanWeissman2019_filtered.h5ad"
DEFAULT_GUIDES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/"
    "GSE133344_filtered_cell_identities.csv.gz"
)
DEFAULT_GUIDES_FILE = "data/raw/GSE133344_filtered_cell_identities.csv.gz"
ACCESSION = "GSE133344"
LICENSE = "CC-BY-4.0"


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the MD5 hex digest of a file, streaming in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest`` (via a .part temp file) with progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {dest.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB",
                        end="",
                        flush=True,
                    )
    print()
    tmp.replace(dest)
    return dest


def ensure_file(url: str, dest: Path, expected_md5=None, force: bool = False):
    """Download ``url`` to ``dest`` unless a valid copy already exists.

    Returns ``(path, downloaded)`` where ``downloaded`` is True if a fresh
    download happened. Raises RuntimeError on checksum mismatch.
    """
    if dest.exists() and not force:
        if expected_md5 is None:
            print(f"  [skip] {dest} already present (no checksum to verify)")
            return dest, False
        if md5sum(dest) == expected_md5:
            print(f"  [skip] {dest} already present (md5 ok)")
            return dest, False
        print(f"  [redo] {dest} exists but md5 mismatch; re-downloading")
    print(f"  [get ] {url}\n         -> {dest}")
    download_file(url, dest)
    if expected_md5 is not None:
        actual = md5sum(dest)
        if actual != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {dest}: expected {expected_md5}, got {actual}"
            )
        print(f"  [ ok ] md5 verified: {actual}")
    return dest, True


def update_config(config_path: Path, raw_file: str, source_url: str, md5: str) -> None:
    """Record accession, source URL, license and download date in the config."""
    with config_path.open() as fh:
        config = yaml.safe_load(fh)
    dataset = config.setdefault("dataset", {})
    dataset["accession"] = ACCESSION
    dataset["source_url"] = source_url
    dataset["license"] = LICENSE
    dataset["md5"] = md5
    dataset["raw_file"] = raw_file
    dataset["download_date"] = date.today().isoformat()
    with config_path.open("w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="configs/data.yaml")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files exist"
    )
    parser.add_argument(
        "--skip-guide-annotations",
        action="store_true",
        help="do not download the GEO per-cell guide-identity CSV",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg: dict = {}
    if config_path.exists():
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}
    dataset_cfg = cfg.get("dataset", {})

    source_url = dataset_cfg.get("source_url", DEFAULT_URL)
    expected_md5 = dataset_cfg.get("md5", DEFAULT_MD5)
    raw_file = dataset_cfg.get("raw_file", DEFAULT_RAW_FILE)

    print("Downloading Norman et al. 2019 Perturb-seq (GEO GSE133344, scPerturb mirror)")
    raw_path, _ = ensure_file(
        source_url,
        PROJECT_ROOT / raw_file,
        expected_md5=expected_md5,
        force=args.force,
    )

    if not args.skip_guide_annotations:
        guides_url = dataset_cfg.get("guide_identities_url", DEFAULT_GUIDES_URL)
        guides_file = dataset_cfg.get("guide_identities_file", DEFAULT_GUIDES_FILE)
        ensure_file(guides_url, PROJECT_ROOT / guides_file, force=args.force)

    update_config(config_path, raw_file, source_url, expected_md5)
    print(
        f"Updated {config_path} "
        f"(accession={ACCESSION}, download_date={date.today().isoformat()})"
    )
    print(f"Done. Raw dataset at: {raw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

