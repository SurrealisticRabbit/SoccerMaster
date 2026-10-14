import hashlib
import logging
import os
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


def download_file(url, local_filename, md5=None, timeout=(15, 60)):
    """Stream a file to disk, verifying an optional md5.

    ``timeout`` is (connect, read) and matters more than it looks: ``requests`` has no
    default, so ``iter_content`` on a stalled connection blocks *forever*. Because the
    progress bar below writes to stderr, and a caller that captures stderr hides it, a
    stalled download presents as a process doing nothing at all indefinitely -- no CPU,
    no memory growth, no output -- which is indistinguishable from a hang. The read
    timeout is per-chunk and generous, so a merely slow connection still completes.

    Downloads land in ``<name>.part`` and are moved into place only once verified, so a
    file at the final path is either complete or absent. That matters because the cache
    is a persistent volume: an interrupted run used to leave a truncated file behind,
    and the md5 check then *refused* to proceed, permanently poisoning the volume --
    every later run failed identically and the only recovery was deleting the file by
    hand. A mismatch on an existing file now discards and refetches it instead.
    """
    local = Path(local_filename)

    if local.exists():
        if md5 is None:
            return local_filename
        if check_md5(local, md5):
            return local_filename
        # Almost always an earlier download that was interrupted. Self-heal rather than
        # raise: the alternative is a cache the user cannot fix without knowing which
        # file to remove and why.
        log.warning(
            "md5 mismatch for existing %s; discarding it and downloading again", local
        )
        local.unlink()

    local.parent.mkdir(exist_ok=True, parents=True)
    # A distinct suffix so a partial file can never be mistaken for a complete one.
    partial = local.with_name(local.name + ".part")
    file_hash = hashlib.md5()

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 8192
            with (open(partial, 'wb') as f,
                  tqdm(desc=f"Downloading {local.name}", total=total_size, unit="B", unit_scale=True) as progress_bar):
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:  # keep-alive chunks arrive as b""
                        continue
                    file_hash.update(chunk)
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        if md5 is not None and md5 != file_hash.hexdigest():
            raise ValueError(
                f'MD5 checksum mismatch when downloading file from {url}. '
                f'Expected {md5}, got {file_hash.hexdigest()}.'
            )
        # Atomic on POSIX and Windows alike, so a concurrent reader never sees a
        # half-written checkpoint at the real path.
        os.replace(partial, local)
    except BaseException:
        # Never leave the partial behind: it would occupy space and, worse, look like
        # a candidate to a later run that only checks for existence.
        partial.unlink(missing_ok=True)
        raise

    return local_filename


def check_md5(local_filename, md5):
    with open(local_filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest() == md5
