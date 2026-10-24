"""
scripts/download_test_video.py
--------------------------------
Downloads a royalty-free restaurant / indoor crowd video for testing.

Uses a short clip from Pexels (public domain / free license).
The clip shows people at tables from an overhead-ish angle.

Run once before starting the pipeline:
    python scripts/download_test_video.py
"""

import sys
import urllib.request
from pathlib import Path


# Pexels free video — people sitting at restaurant/cafe tables
# License: Pexels License (free to use for testing)
VIDEO_URL = (
    "https://videos.pexels.com/video-files/3209828/3209828-uhd_2560_1440_25fps.mp4"
)
FALLBACK_URL = (
    "https://videos.pexels.com/video-files/856377/856377-hd_1920_1080_25fps.mp4"
)
OUTPUT_PATH = Path("data/restaurant_sample.mp4")


def download(url: str, dest: Path) -> bool:
    print(f"Downloading: {url}")
    print(f"  → {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response, open(dest, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 256  # 256 KB chunks
            while True:
                data = response.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}% ({downloaded // 1024 // 1024} MB)", end="", flush=True)
        print(f"\nDone! Saved to: {dest}")
        return True
    except Exception as exc:
        print(f"\nFailed: {exc}")
        return False


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
        print(f"Video already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        print("Delete it and re-run to re-download.")
        return

    ok = download(VIDEO_URL, OUTPUT_PATH)
    if not ok:
        print("Primary URL failed. Trying fallback…")
        ok = download(FALLBACK_URL, OUTPUT_PATH)

    if not ok:
        print(
            "\nAutomatic download failed.\n"
            "Please download any MP4 restaurant video manually and save it to:\n"
            f"  {OUTPUT_PATH.resolve()}\n"
            "Then run the pipeline with:\n"
            "  python -m cv_worker.main --video data/restaurant_sample.mp4"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
