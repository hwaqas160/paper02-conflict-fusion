"""Minimal PCD reader (ascii / binary / binary_compressed) for MAN TruckScenes.

binary_compressed = [uint32 compressed size][uint32 raw size][LZF stream], raw data laid out field-major
(all x, then all y, ...). LZF decoding is JIT-compiled with numba (pure Python would be ~100x slower).
"""
import numba
import numpy as np

_NP = {("F", 4): np.float32, ("F", 8): np.float64, ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32,
       ("U", 8): np.uint64, ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32, ("I", 8): np.int64}


@numba.njit(cache=True)
def _lzf_decompress(src, dst):
    ip, op, n_in, n_out = 0, 0, src.size, dst.size
    while ip < n_in:
        ctrl = src[ip]
        ip += 1
        if ctrl < 32:
            ln = ctrl + 1
            if op + ln > n_out or ip + ln > n_in:
                return -1
            for k in range(ln):
                dst[op + k] = src[ip + k]
            op += ln
            ip += ln
        else:
            ln = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1
            if ln == 7:
                ln += src[ip]
                ip += 1
            ref -= src[ip]
            ip += 1
            ln += 2
            if op + ln > n_out or ref < 0:
                return -1
            for k in range(ln):
                dst[op + k] = dst[ref + k]
            op += ln
    return op


def read_pcd(path):
    """-> dict field_name -> 1-D numpy array."""
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.find(b"DATA")
    nl = raw.find(b"\n", end)
    hdr = {}
    for line in raw[:nl].decode("ascii").splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            hdr[parts[0]] = parts[1:]
    fields, sizes, types = hdr["FIELDS"], list(map(int, hdr["SIZE"])), hdr["TYPE"]
    counts = list(map(int, hdr.get("COUNT", ["1"] * len(fields))))
    n = int(hdr["POINTS"][0])
    mode = hdr["DATA"][0]
    body = raw[nl + 1:]
    dts = [_NP[(t, s)] for t, s in zip(types, sizes)]
    if mode == "ascii":
        arr = np.loadtxt(body.decode("ascii").splitlines(), ndmin=2)
        return {f: arr[:, i].astype(dt) for i, (f, dt) in enumerate(zip(fields, dts))}
    if mode == "binary":
        dt = np.dtype([(f, d, (c,)) if c > 1 else (f, d) for f, d, c in zip(fields, dts, counts)])
        rec = np.frombuffer(body[:n * dt.itemsize], dtype=dt)
        return {f: np.asarray(rec[f]) for f in fields}
    if mode == "binary_compressed":
        csize, usize = np.frombuffer(body[:8], dtype=np.uint32)
        out = np.empty(int(usize), np.uint8)
        got = _lzf_decompress(np.frombuffer(body[8:8 + int(csize)], dtype=np.uint8), out)
        if got != usize:
            raise IOError(f"LZF decode failed for {path} ({got} != {usize})")
        res, off = {}, 0
        for f, d, s, c in zip(fields, dts, sizes, counts):
            nb = n * s * c
            res[f] = np.frombuffer(out[off:off + nb].tobytes(), dtype=d)
            off += nb
        return res
    raise ValueError(f"unsupported PCD DATA mode {mode}")
