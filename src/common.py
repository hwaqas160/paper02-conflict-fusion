"""Config, paths, seeding, device selection, and small I/O helpers shared by every module."""
import hashlib
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("camera", "lidar", "radar")
CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_config(path=None, overrides=None):
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def resolve(cfg, key):
    """Absolute path for cfg['paths'][key]."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device(pref="auto", min_free_gb=2.0):
    import torch
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 2**30 >= min_free_gb:
            return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name, log_file=None):
    """Named loggers are process-wide singletons in the `logging` module: a second call with the same
    `name` but a different `log_file` (e.g. one Ctx per multiseed seed, each with its own results_root)
    would silently keep logging to the FIRST call's file without this check, since `logging.getLogger`
    returns the same object every time. Add a FileHandler for any log_file not already attached, instead
    of short-circuiting on "has any handler at all"."""
    logger = logging.getLogger(name)
    if log_file:
        log_file = str(Path(log_file).resolve())
        if any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logger.handlers):
            return logger
    elif logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S")
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def source_tree_hash(root=None):
    """SHA-256 over src/ and configs/ — stands in for a git SHA (this workspace is not a git repo)."""
    root = Path(root) if root else PROJECT_ROOT
    h = hashlib.sha256()
    for sub in ("src", "configs"):
        for p in sorted((root / sub).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".yaml"} and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(root)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()[:16]


def condition_tag(condition, severity):
    return f"{condition}{severity}"


def iter_conditions(cfg):
    for cond, sevs in cfg["weather"]["conditions"].items():
        for s in sevs:
            yield cond, int(s)


def env_threads(n):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
