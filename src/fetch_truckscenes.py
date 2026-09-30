"""MAN TruckScenes v1.2 (public S3, CC BY-NC-SA 4.0) selective download for Registration 2.

The sensor data ships as seven ~80 GB zips. We read each zip's central directory once over HTTPS, then pull
only the members we use -- labelled keyframes of the forward-facing sensors, plus each radar's previous sweep
(temporal persistence) -- with one HTTP range request per file. ~2.6 MB per keyframe instead of ~24 MB.

    python -m src.fetch_truckscenes index                  # central directories -> data/truckscenes/zip_index.pkl
    python -m src.fetch_truckscenes fetch --workers 32     # all trainval keyframes of the chosen channels
"""
import argparse
import io
import json
import pickle
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import PROJECT_ROOT, get_logger
from .fetch_cadc import HttpFile

BASE = "https://man-truckscenes.s3.amazonaws.com/release/trainval/"
ZIPS = [f"man-truckscenes_sensordata0{i}_v1.2-trainval.zip" for i in range(1, 8)]
ROOT = PROJECT_ROOT / "data" / "truckscenes"
META = ROOT / "man-truckscenes" / "v1.2-trainval"
LIDARS = ["LIDAR_TOP_FRONT", "LIDAR_LEFT", "LIDAR_RIGHT"]
CAMERA = "CAMERA_LEFT_FRONT"
RADARS = ["RADAR_LEFT_FRONT", "RADAR_RIGHT_FRONT"]
CHANNELS = LIDARS + [CAMERA] + RADARS


def _get(url, rng, retries=6):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 * (a + 1))


def build_index(log):
    idx = {}
    for zi, name in enumerate(ZIPS):
        t0 = time.time()
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(BASE + name), 8 << 20))
        n = 0
        for i in z.infolist():
            fn = i.filename.split("man-truckscenes/", 1)[-1]
            if fn.startswith("samples/") or (fn.startswith("sweeps/RADAR_") and any(r in fn for r in RADARS)):
                idx[fn] = (zi, i.header_offset, i.compress_size, i.file_size, i.compress_type, len(i.filename))
                n += 1
        log.info(f"{name}: {n} members indexed ({time.time() - t0:.0f}s)")
    with open(ROOT / "zip_index.pkl", "wb") as f:
        pickle.dump(idx, f)
    return idx


def wanted_files(log):
    """Keyframe files of CHANNELS for every trainval sample, plus the previous sweep of each radar."""
    sensors = {s["token"]: s["channel"] for s in json.load(open(META / "sensor.json"))}
    calib = {c["token"]: sensors[c["sensor_token"]] for c in json.load(open(META / "calibrated_sensor.json"))}
    log.info("loading sample_data.json ...")
    sd = json.load(open(META / "sample_data.json"))
    by_tok = {r["token"]: r for r in sd}
    out = set()
    for r in sd:
        if not r["is_key_frame"]:
            continue
        ch = calib[r["calibrated_sensor_token"]]
        if ch not in CHANNELS:
            continue
        out.add(r["filename"])
        if ch in RADARS and r["prev"]:
            out.add(by_tok[r["prev"]]["filename"])
    log.info(f"{len(out)} files wanted")
    return sorted(out)


def fetch_one(fn, entry):
    zi, off, csize, fsize, method, name_len = entry
    dst = ROOT / fn
    if dst.exists() and dst.stat().st_size == fsize:
        return 0
    url = BASE + ZIPS[zi]
    guess = 30 + name_len + 256
    buf = _get(url, (off, off + guess + csize - 1))
    if buf[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {fn}")
    n, e = int.from_bytes(buf[26:28], "little"), int.from_bytes(buf[28:30], "little")
    start = 30 + n + e
    if start + csize > len(buf):  # extra field longer than guessed
        buf = _get(url, (off, off + start + csize - 1))
    data = buf[start:start + csize]
    if method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif method != zipfile.ZIP_STORED:
        raise IOError(f"unsupported compression {method} for {fn}")
    if len(data) != fsize:
        raise IOError(f"size mismatch for {fn}: {len(data)} != {fsize}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "fetch"])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    log = get_logger("fetch_ts", PROJECT_ROOT / "results" / "logs" / "fetch_truckscenes.log")
    if args.cmd == "index":
        build_index(log)
        return
    with open(ROOT / "zip_index.pkl", "rb") as f:
        idx = pickle.load(f)
    files = wanted_files(log)
    missing = [f for f in files if f not in idx]
    if missing:
        log.warning(f"{len(missing)} wanted files not in any zip, e.g. {missing[:3]}")
    todo = [f for f in files if f in idx]
    t0, done, nbytes, failed = time.time(), 0, 0, 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_one, f, idx[f]): f for f in todo}
        for fut in as_completed(futs):
            try:
                nbytes += fut.result()
            except Exception as e:
                failed += 1
                log.error(f"{futs[fut]}: {e}")
            done += 1
            if done % 2000 == 0 or done == len(todo):
                el = time.time() - t0
                log.info(f"{done}/{len(todo)} files, {nbytes / 1e9:.1f} GB, {nbytes / 1e6 / max(el, 1):.1f} MB/s, "
                         f"{failed} failed, ETA {(len(todo) - done) * el / max(done, 1) / 3600:.1f} h")


if __name__ == "__main__":
    main()
