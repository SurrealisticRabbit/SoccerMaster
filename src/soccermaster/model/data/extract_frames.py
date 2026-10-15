"""
Extract frames from Soccer Factory video packages and restore the original image sequence directory structure.

Input (after extracting tar files):
    videos/
    ├── SNGS-10001.mp4
    ├── SNGS-10002.mp4
    └── ...

Output (restored directory structure expected by training code):
    output_dir/
    ├── SNGS-10001/
    │   └── img1/
    │       ├── 000001.jpg
    │       ├── 000002.jpg
    │       └── ...
    ├── SNGS-10002/
    │   └── img1/
    │       └── ...
    └── ...

Usage:
    # 1. Extract tar files first, then decode
    tar xf soccer_factory_videos_part1.tar
    tar xf soccer_factory_videos_part2.tar
    ...

    # 2. Decode videos to image sequences
    python data/extract_frames.py \
        --video_dir ./videos \
        --output_dir ./datasets/SoccerNetGS/sn500 \
        --quality 95

    # Or extract directly from tar files (no intermediate extraction needed):
    python data/extract_frames.py \
        --tar_files soccer_factory_videos_part1.tar soccer_factory_videos_part2.tar \
        --output_dir ./datasets/SoccerNetGS/sn500 \
        --quality 95

Dependencies:
    - ffmpeg (system installation)

Note:
    - Since the video encoding is lossy (H.264 CRF=18), restored jpg images have minor differences from the originals.
    - The differences are imperceptible to the human eye and do not affect training performance.
    - Default output jpg quality=95, adjustable via --quality parameter.
"""

import os
import sys
import glob
import argparse
import subprocess
import tarfile
import tempfile
import shutil

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def probe_duration(video_path: str) -> float | None:
    """Return the video's duration in seconds via ffprobe, or None if unavailable.

    Used to size the ffmpeg timeout. A single fixed timeout cannot work here, because
    decoding cost is proportional to video length: any constant is both too small for a
    full match and pointlessly large for a short clip.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return float(result.stdout.strip())
    except Exception:  # noqa: BLE001 - ffprobe may be absent; fall back to the default
        return None


#: Keys ffmpeg emits on the ``-progress`` stream. Anything else on the merged stream is
#: an error message and gets printed verbatim.
_FFMPEG_PROGRESS_KEYS = {
    "frame", "fps", "bitrate", "total_size", "out_time_us", "out_time_ms",
    "out_time", "dup_frames", "drop_frames", "speed", "progress",
}


def _stream_ffmpeg(cmd, timeout, expected_frames=None, report_every=5.0):
    """Run ffmpeg, echoing errors and summarising its ``-progress`` stream.

    Returns ``(returncode, tail)`` where tail is the last of the output, for the error
    path. Progress lines are summarised rather than printed raw because ffmpeg emits a
    block of key=value pairs several times a second, which is noise rather than status.

    Timeout is enforced here because ``Popen`` has no ``timeout`` argument; the process
    is killed and the returncode reported as non-zero so the caller's existing failure
    path handles it.
    """
    import time
    from collections import deque

    started = time.time()
    last_report = 0.0
    fields: dict[str, str] = {}
    tail: deque[str] = deque(maxlen=40)

    # stderr folded into stdout: one pipe to drain, so neither can fill and block.
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    def report() -> None:
        frame = fields.get("frame", "?")
        speed = fields.get("speed", "")
        elapsed = time.time() - started
        total = f"/{expected_frames}" if expected_frames else ""
        percent = ""
        if expected_frames and fields.get("frame", "").isdigit():
            percent = f" ({100 * int(fields['frame']) / expected_frames:4.1f}%)"
        print(
            f"  decode  {frame}{total} frames{percent}  {speed}  {elapsed:,.0f}s",
            flush=True,
        )

    assert process.stdout is not None
    try:
        for raw in process.stdout:
            line = raw.rstrip("\n")
            tail.append(line)
            key, sep, value = line.partition("=")
            if sep and key in _FFMPEG_PROGRESS_KEYS:
                fields[key] = value.strip()
                if key == "progress":  # end of one update block
                    now = time.time()
                    if now - last_report >= report_every:
                        last_report = now
                        report()
                continue
            if line.strip():
                # An error or warning: show it as it happens rather than only on failure.
                print(f"  ffmpeg: {line}", flush=True)

            if time.time() - started > timeout:
                process.kill()
                process.wait(timeout=30)
                return 1, f"ffmpeg exceeded {timeout:.0f}s"
    except KeyboardInterrupt:
        process.terminate()
        raise
    finally:
        process.wait()

    if fields.get("frame"):
        report()
    return process.returncode, "\n".join(tail)


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    quality: int = 95,
    fps: float | None = None,
    max_frames: int | None = None,
    timeout: float | None = None,
):
    """
    Use ffmpeg to extract frames from a video as jpg images.

    Args:
        video_path: path to mp4 video file
        output_dir: output directory (will create img1/ subdirectory)
        quality: jpg quality (1-100, default 95)
        fps: keep only this many frames per second. ``None`` keeps every frame, which is
            what the game-state steps were built against, but it makes cost scale
            linearly with video length -- an 11-minute match at 25 fps is ~16,500 frames,
            and step 3 asks a 7B VLM about every player in every one of them.
        max_frames: stop after this many output frames. This is the lever for testing at
            **native frame rate** on a long video. ``fps`` cannot do that: an ffmpeg
            ``fps`` filter still decodes every input frame and merely discards the ones
            it does not want, so decimation costs the same to decode and only saves the
            downstream steps -- while making the tracker's job unrealistically hard,
            since it assumes small motion between consecutive frames. ``max_frames``
            instead takes a short genuine slice of the video, so a few hundred frames
            exercise the real thing.
        timeout: seconds to allow ffmpeg. ``None`` derives it from the video's duration.
            This was hardcoded to 300, which silently failed on any video needing more
            than five minutes of decoding and left a half-populated img1/ behind.

    Returns:
        dict with num_frames, success status
    """
    img1_dir = os.path.join(output_dir, "img1")
    os.makedirs(img1_dir, exist_ok=True)

    # ffmpeg: 视频 -> 图片序列
    # 输出格式: %06d.jpg (000001.jpg, 000002.jpg, ...)
    # -qscale:v 2 对应约 quality=95 的 jpg
    # jpg quality mapping: ffmpeg uses qscale 2-31 (2=best, 31=worst)
    # Approximate: quality 95 -> qscale 2, quality 85 -> qscale 5, quality 75 -> qscale 8
    qscale = max(2, min(31, int(2 + (100 - quality) * 29 / 100)))

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', video_path]

    if fps is not None:
        # -vf fps=N keeps/drops frames to hit N per second, applied before the encode.
        cmd += ['-vf', f'fps={fps}']

    if max_frames is not None:
        # An OUTPUT option, so it has to precede the output pattern.
        cmd += ['-frames:v', str(max_frames)]

    cmd += ['-qscale:v', str(qscale), '-start_number', '1']
    # Machine-readable progress on stdout, newline-delimited key=value pairs. ffmpeg's
    # human-readable stats are carriage-return-updated, which a pipe collapses into one
    # unreadable line -- and until now the whole of ffmpeg's output was captured and
    # discarded, so decoding a long video was a silent black box.
    cmd += ['-progress', 'pipe:1', '-nostats']
    cmd.append(os.path.join(img1_dir, '%06d.jpg'))

    duration = probe_duration(video_path)
    if timeout is None:
        # Generous on purpose: decode throughput varies wildly with resolution and
        # codec, and being too tight leaves a half-populated img1/ that then
        # masquerades as a pipeline bug. Ten times realtime, floored at ten minutes.
        timeout = max(600.0, duration * 10) if duration else 3600.0

    # Estimate the frame count so the progress line can show a percentage. A hint only:
    # the real count comes from what ffmpeg actually writes.
    expected = None
    if duration and fps:
        expected = max(1, int(duration * fps))
    if expected and max_frames:
        expected = min(expected, max_frames)

    try:
        returncode, tail = _stream_ffmpeg(cmd, timeout, expected_frames=expected)
        if returncode != 0:
            return {
                "success": False,
                "reason": f"ffmpeg error: {tail[-2000:]}",
                "num_frames": 0,
            }
    except FileNotFoundError:
        return {"success": False, "reason": "ffmpeg not found", "num_frames": 0}

    # 统计输出帧数
    num_frames = len([f for f in os.listdir(img1_dir) if f.endswith('.jpg')])

    return {"success": True, "num_frames": num_frames}


def process_video_dir(video_dir: str, output_dir: str, quality: int = 95):
    """
    Process a directory containing mp4 files, extracting frames from each into the corresponding sequence directory.
    """
    mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    if not mp4_files:
        print(f"Error: no .mp4 files found in {video_dir}")
        return

    print(f"Found {len(mp4_files)} video files")
    print(f"Output directory: {output_dir}")
    print(f"JPEG quality: {quality}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_frames = 0

    for mp4_path in tqdm(mp4_files, desc="Extracting"):
        video_id = os.path.splitext(os.path.basename(mp4_path))[0]
        seq_output_dir = os.path.join(output_dir, video_id)

        img1_dir = os.path.join(seq_output_dir, "img1")
        if os.path.exists(img1_dir) and len(os.listdir(img1_dir)) > 0:
            skip_count += 1
            continue

        result = extract_frames_from_video(mp4_path, seq_output_dir, quality=quality)

        if result["success"]:
            success_count += 1
            total_frames += result["num_frames"]
        else:
            fail_count += 1
            print(f"\n  Failed: {video_id} - {result['reason']}")

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"{'='*60}")
    print(f"  Success: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output directory: {output_dir}")


def process_tar_files(tar_files: list, output_dir: str, quality: int = 95):
    """
    Extract mp4 files directly from tar archives and decode frames, without extracting the full tar first.
    """
    print(f"Extracting and decoding from {len(tar_files)} tar files...")
    print(f"Output directory: {output_dir}")
    print(f"JPEG quality: {quality}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_frames = 0

    for tar_path in tar_files:
        if not os.path.exists(tar_path):
            print(f"Warning: tar file not found, skipping: {tar_path}")
            continue

        print(f"\nProcessing: {tar_path}")

        with tarfile.open(tar_path, 'r') as tar:
            mp4_members = [m for m in tar.getmembers() if m.name.endswith('.mp4')]
            print(f"  Contains {len(mp4_members)} videos")

            for member in tqdm(mp4_members, desc=f"  {os.path.basename(tar_path)}"):
                # Get sequence ID: videos/SNGS-10001.mp4 -> SNGS-10001
                video_id = os.path.splitext(os.path.basename(member.name))[0]
                seq_output_dir = os.path.join(output_dir, video_id)

                # Skip already extracted sequences (resume support)
                img1_dir = os.path.join(seq_output_dir, "img1")
                if os.path.exists(img1_dir) and len(os.listdir(img1_dir)) > 0:
                    skip_count += 1
                    continue

                # 提取 mp4 到临时文件
                tmp_fd, tmp_mp4 = tempfile.mkstemp(suffix='.mp4')
                os.close(tmp_fd)
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        fail_count += 1
                        continue
                    with open(tmp_mp4, 'wb') as out:
                        out.write(f.read())

                    result = extract_frames_from_video(tmp_mp4, seq_output_dir, quality=quality)

                    if result["success"]:
                        success_count += 1
                        total_frames += result["num_frames"]
                    else:
                        fail_count += 1
                        print(f"\n    Failed: {video_id} - {result['reason']}")
                finally:
                    os.remove(tmp_mp4)

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"{'='*60}")
    print(f"  Success: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from Soccer Factory video packages to restore image sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Option 1: Extract tar files first, then decode
  tar xf soccer_factory_videos_part1.tar
  python data/extract_frames.py --video_dir ./videos --output_dir ./sn500

  # Option 2: Decode directly from tar files (no intermediate extraction)
  python data/extract_frames.py \\
      --tar_files soccer_factory_videos_part*.tar \\
      --output_dir ./sn500
        """
    )
    parser.add_argument(
        '--video_dir', type=str, default=None,
        help='Directory containing extracted .mp4 files'
    )
    parser.add_argument(
        '--tar_files', type=str, nargs='+', default=None,
        help='Tar file paths (one or more, extracts directly from tar)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for restored image sequences '
             '(default: <data_dir>/SoccerNetGS/sn500)'
    )
    parser.add_argument(
        '--quality', type=int, default=95,
        help='Output JPEG quality (1-100, default: 95)'
    )
    args = parser.parse_args()

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: ffmpeg is not installed or not available! Please install ffmpeg first.")
        sys.exit(1)

    if args.video_dir and args.tar_files:
        print("Error: --video_dir and --tar_files cannot be specified at the same time")
        sys.exit(1)

    if not args.video_dir and not args.tar_files:
        print("Error: must specify either --video_dir or --tar_files")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        from soccermaster import paths

        output_dir = str(paths.data_dir() / "SoccerNetGS" / "sn500")

    if args.video_dir:
        process_video_dir(args.video_dir, output_dir, quality=args.quality)
    else:
        process_tar_files(args.tar_files, output_dir, quality=args.quality)


if __name__ == '__main__':
    main()
