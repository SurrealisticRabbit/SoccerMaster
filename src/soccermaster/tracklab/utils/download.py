import hashlib
import logging
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


#: 1 MiB. The original 8 KiB meant a Python-level iteration, a syscall and a tqdm update
#: every 8 KB -- roughly 48,000 of them for a 378 MiB checkpoint. That is wasteful rather
#: than necessarily fatal, but it is the first thing to remove when throughput is poor.
CHUNK_SIZE = 1 << 20


def _hash_file(path, chunk_size: int = CHUNK_SIZE):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest


def download_file(url, local_filename, md5=None, timeout=(15, 120), resume=True,
                  attempts=6, retry_backoff=5.0):
    """Stream a file to disk, verifying an optional md5, resuming and retrying.

    Three properties this needs, each learned the hard way:

    **Atomicity.** The download goes to ``<name>.part`` and is moved into place with
    ``os.replace`` only once verified, so a file at the final path is either complete or
    absent. An earlier version wrote straight to the target, so an interrupted run left a
    truncated file that then failed its md5 check -- and because the cache is a persistent
    volume, that failure was permanent and needed manual deletion.

    **Resume.** When the server honours ``Range`` (Zenodo does, without advertising
    ``Accept-Ranges``) an existing ``.part`` is continued rather than restarted. Observed
    in practice: a 396 MB checkpoint arriving at ~110 KB/s over a link that dropped after
    28 MB. Without resume every retry pays the whole transfer again, so the download never
    finishes. If the server answers a ranged request with 200 instead of 206 it has
    ignored the range, so the partial is discarded and the transfer restarts rather than
    being appended to -- appending there would silently corrupt the file.

    **Retry.** A dropped connection raises ``ChunkedEncodingError``/``IncompleteRead``,
    which is a transient fault on a long slow transfer rather than a reason to fail the
    run. Retries resume from where they stopped, so progress accumulates instead of being
    lost.

    ``timeout`` is (connect, read) and matters more than it looks: ``requests`` has no
    default, so ``iter_content`` on a stalled connection blocks *forever*, and the progress
    bar writes to stderr, which a caller that captures stderr hides -- so a stall presents
    as a process doing nothing at all, with no CPU, no memory growth and no output.
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
    partial = local.with_name(local.name + ".part")

    for attempt in range(1, attempts + 1):
        done = partial.stat().st_size if (resume and partial.exists()) else 0
        file_hash = _hash_file(partial) if done else hashlib.md5()
        if done:
            log.info("resuming %s at %d bytes", local.name, done)

        headers = {"Range": f"bytes={done}-"} if done else {}

        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if done and r.status_code == 200:
                    # The server ignored Range and is sending the whole body from the
                    # start. Appending would corrupt the file, so start over deliberately.
                    log.warning("server ignored the range request; restarting %s", local.name)
                    done = 0
                    file_hash = hashlib.md5()

                r.raise_for_status()
                remaining = int(r.headers.get("content-length", 0))
                total = (done + remaining) if remaining else (done or None)

                with (open(partial, "ab" if done else "wb") as f,
                      tqdm(desc=f"Downloading {local.name}", total=total, initial=done,
                           unit="B", unit_scale=True) as progress_bar):
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:  # keep-alive chunks arrive as b""
                            continue
                        file_hash.update(chunk)
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            if md5 is not None and md5 != file_hash.hexdigest():
                # The bytes are wrong rather than merely incomplete, so the partial is
                # useless -- most likely a corrupt prefix left by an earlier attempt that
                # resume then appended to. Discard it and fetch the whole file again.
                partial.unlink(missing_ok=True)
                if attempt == attempts:
                    raise ValueError(
                        f"MD5 checksum mismatch for {local.name} after {attempts} "
                        f"attempt(s): expected {md5}, got {file_hash.hexdigest()}."
                    )
                log.warning(
                    "md5 mismatch for %s on attempt %d; refetching from scratch",
                    local.name, attempt,
                )
                continue

            # Atomic on POSIX and Windows alike, so a concurrent reader never sees a
            # half-written checkpoint at the real path.
            os.replace(partial, local)
            return local_filename

        except KeyboardInterrupt:
            # Leave the partial so a later run can resume it, and do not retry.
            raise
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - network faults are the expected case
            if attempt == attempts:
                raise
            wait = min(60.0, retry_backoff * attempt)
            log.warning(
                "download of %s failed on attempt %d/%d (%s); resuming in %.0fs",
                local.name, attempt, attempts, type(exc).__name__, wait,
            )
            if wait:
                time.sleep(wait)

    raise RuntimeError(f"download of {local.name} failed without an exception")  # pragma: no cover


def check_md5(local_filename, md5):
    return _hash_file(local_filename).hexdigest() == md5
