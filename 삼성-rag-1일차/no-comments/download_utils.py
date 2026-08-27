from pathlib import Path

import requests


def download_if_missing(url, destination, timeout=60):
    destination = Path(destination)
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_file = destination.with_suffix(destination.suffix + ".part")

    try:
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with partial_file.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        partial_file.replace(destination)
    except Exception:
        partial_file.unlink(missing_ok=True)
        raise

    return destination
