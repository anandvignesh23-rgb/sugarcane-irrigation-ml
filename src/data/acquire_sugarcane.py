from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import requests


class DatasetAcquisitionError(RuntimeError):
    """Raised when the exact required SIDSS sugarcane dataset cannot be acquired."""


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_primary_dataset(dataset_config: dict[str, Any]) -> str:
    return (
        f"{dataset_config.get('name')} | "
        f"Institution/source association: {dataset_config.get('institution_source_association')} | "
        f"Role: {dataset_config.get('role')}"
    )


def ensure_sidss_sugarcane_dataset(target_path: str | Path, dataset_config: dict[str, Any]) -> Path:
    """Ensure the required NIT Patna/SIDSS sugarcane dataset exists locally.

    The function only attempts the configured exact dataset URLs and never falls back
    to an unrelated dataset.
    """

    path = Path(target_path)
    expected_name = dataset_config.get("name", "Sugarcane Irrigation Dataset for the SIDSS Framework")
    expected_md5 = dataset_config.get("expected_md5")
    if path.exists():
        if expected_md5 and _md5(path) != expected_md5:
            raise DatasetAcquisitionError(
                f"Existing dataset at {path} does not match the expected MD5 for "
                f"{expected_name}. Remove the file or update config only after verifying "
                "that it is the exact NIT Patna/SIDSS dataset."
            )
        return path

    urls = dataset_config.get("download_urls") or []
    if not urls:
        raise DatasetAcquisitionError(
            f"{expected_name} is mandatory, but no download URLs are configured."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            temporary.write_bytes(response.content)
            if expected_md5 and _md5(temporary) != expected_md5:
                errors.append(f"{url}: downloaded file MD5 did not match expected value")
                temporary.unlink(missing_ok=True)
                continue
            temporary.replace(path)
            return path
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
        finally:
            temporary.unlink(missing_ok=True)

    joined_errors = "; ".join(errors) if errors else "no configured URLs were attempted"
    raise DatasetAcquisitionError(
        f"Could not retrieve the exact required dataset: {describe_primary_dataset(dataset_config)}. "
        f"Tried only the configured SIDSS/NIT Patna sources. Details: {joined_errors}"
    )
