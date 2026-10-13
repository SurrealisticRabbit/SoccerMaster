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


def extract_frames_from_video(video_path: str, output_dir: str, quality: int = 95):
    """
    Use ffmpeg to extract all frames from a video as jpg images.

    Args:
        video_path: path to mp4 video file
        output_dir: output directory (will create img1/ subdirectory)
        quality: jpg quality (1-100, default 95)

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

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-qscale:v', str(qscale),
        '-start_number', '1',
        os.path.join(img1_dir, '%06d.jpg')
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"success": False, "reason": f"ffmpeg error: {result.stderr[-200:]}", "num_frames": 0}
    except subprocess.TimeoutExpired:
        return {"success": False, "reason": "ffmpeg timeout", "num_frames": 0}
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
        '--output_dir', type=str, required=True,
        help='Output directory for restored image sequences'
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

    if args.video_dir:
        process_video_dir(args.video_dir, args.output_dir, quality=args.quality)
    else:
        process_tar_files(args.tar_files, args.output_dir, quality=args.quality)


if __name__ == '__main__':
    main()
