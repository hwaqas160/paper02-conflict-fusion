"""CADC selective download (Exp 5). The Waterloo server is public (no registration, verified 2026-09-21).

Only what the cache uses is fetched: per-date calib, per-drive 3d_ann.json, and from each drive's
labeled.zip ONLY labeled/image_00 (front camera) and labeled/lidar_points. Members are read straight
out of the remote zip with HTTP range requests, so the other seven cameras are never downloaded
(~4x less traffic than the devkit's full-zip download). Resumable: files already on disk are skipped.

    python -m src.fetch_cadc                      # all 75 drives
    python -m src.fetch_cadc --drives 2018_03_06/0001 2019_02_27/0002
"""
import argparse
import io
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import get_logger, load_config, resolve

BASE = "http://wiselab.uwaterloo.ca/cadcd_data/"
DATES = ("2018_03_06", "2018_03_07", "2019_02_27")
KEEP = ("labeled/image_00/", "labeled/lidar_points/")


def _get(url, headers=None, retries=4):
    import time
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP Range (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = _get(self.url, {"Range": f"bytes={self.pos}-{end}"})
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def list_drives():
    out = []
    for d in DATES:
        html = _get(BASE + d + "/").decode("utf-8", "ignore")
        out += [f"{d}/{x}" for x in sorted(set(re.findall(r'href="(\d{4})/"', html)))]
    return out


def fetch_calib(root, date):
    dst = root / date / "calib"
    if (dst / "extrinsics.yaml").exists():
        return
    z = zipfile.ZipFile(io.BytesIO(_get(BASE + date + "/calib.zip")))
    for n in z.namelist():
        if n.endswith("/"):
            continue
        p = root / date / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(z.read(n))


def fetch_drive(root, drive):
    date, num = drive.split("/")
    ddir = root / date / num
    ddir.mkdir(parents=True, exist_ok=True)
    ann = ddir / "3d_ann.json"
    if not ann.exists():
        ann.write_bytes(_get(BASE + drive + "/3d_ann.json"))
    done_flag = ddir / ".fetched"
    if done_flag.exists():
        return drive, 0
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(BASE + drive + "/labeled.zip"), 4 << 20))
    n_new = 0
    for info in z.infolist():
        if info.is_dir() or not info.filename.startswith(KEEP):
            continue
        p = ddir / info.filename
        if p.exists() and p.stat().st_size == info.file_size:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(z.read(info))
        tmp.replace(p)
        n_new += 1
    done_flag.write_text("ok")
    return drive, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drives", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config()
    root = resolve(cfg, "cadc_root")
    log = get_logger("fetch_cadc", resolve(cfg, "results_root") / "logs" / "fetch_cadc.log")
    drives = args.drives or list_drives()
    log.info(f"{len(drives)} drives -> {root}")
    for d in sorted({x.split('/')[0] for x in drives}):
        fetch_calib(root, d)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_drive, root, d): d for d in drives}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                d, n = f.result()
                log.info(f"[{i}/{len(drives)}] {d}: {n} files")
            except Exception as e:
                log.error(f"[{i}/{len(drives)}] {futs[f]}: {e}")


if __name__ == "__main__":
    main()
